// ============================================================
//  MAF Learning – 14: Batch Sentiment Analysis
//
//  A real-world batch workload:
//    5,000 Amazon product reviews classified by o4-mini
//    using the Azure OpenAI Batch API.
//
//  Pipeline:
//    1. Download 5,000 reviews from HuggingFace (amazon_polarity)
//    2. Build a .jsonl batch input file (one request per review)
//    3. Upload the file to Azure OpenAI
//    4. Submit a batch job
//    5. Poll until complete
//    6. Download and parse the output .jsonl
//    7. Save results to batch_results.csv
//    8. Print an accuracy report vs. ground-truth labels
//
//  Each review is classified as:
//    sentiment     — positive | negative | neutral
//    has_complaint — true | false
//    category_tag  — electronics | books | music | food |
//                    clothing | home | beauty | sports | other
//
//  Estimated cost at 5,000 rows: ~$0.74 (Batch API pricing)
// ============================================================

using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Configuration;

// ── 1. Load .env then config ──────────────────────────────
DotNetEnv.Env.Load(Path.Combine(AppContext.BaseDirectory, ".env"));

IConfiguration config = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

string endpoint        = config["AzureAI:Endpoint"]        ?? throw new InvalidOperationException("AzureAI:Endpoint missing");
string apiKey          = config["AzureAI:ApiKey"]          ?? throw new InvalidOperationException("AzureAI:ApiKey missing");
string modelDeployment = config["AzureAI:ModelDeployment"] ?? "o4-mini";

Console.WriteLine("==============================================");
Console.WriteLine("  MAF Learning – 14: Batch Sentiment Analysis");
Console.WriteLine("==============================================");
Console.WriteLine($"  Endpoint : {endpoint}");
Console.WriteLine($"  Model    : {modelDeployment}");
Console.WriteLine();

// ── 2. Download reviews from HuggingFace ─────────────────
//
//  We use the fancyzhx/amazon_polarity dataset (400K reviews).
//  Each review comes with a ground-truth label (0=neg, 1=pos)
//  which we use at the end to measure model accuracy.
//
const int ReviewCount = 100; // set to 5_000 after increasing Global Batch token quota in Azure AI Foundry

Console.WriteLine($"Step 1/8 — Downloading {ReviewCount:N0} reviews from HuggingFace...");

using var http = new HttpClient();
http.Timeout = TimeSpan.FromMinutes(3);

List<AmazonReview> reviews = await ReviewLoader.LoadAsync(ReviewCount, http);
Console.WriteLine($"  Downloaded {reviews.Count:N0} reviews.");

// ── 3. Build .jsonl input ─────────────────────────────────
//
//  System prompt asks for structured JSON output with three fields.
//  response_format: json_object ensures the model returns valid JSON.
//  max_tokens: 80 caps output cost (the JSON response is small).
//
Console.WriteLine();
Console.WriteLine("Step 2/8 — Building .jsonl batch input...");

const string SystemPrompt =
    "You are a product review classifier. " +
    "Given a review title and text, return a JSON object with exactly these fields: " +
    "\"sentiment\" (one of: positive, negative, neutral), " +
    "\"has_complaint\" (boolean, true if the review mentions a specific problem or complaint), " +
    "\"category_tag\" (one of: electronics, books, music, food, clothing, home, beauty, sports, other). " +
    "Return only the JSON object, no explanation.";

var jsonlBuilder = new StringBuilder();

foreach (var review in reviews)
{
    string userContent = $"Title: {review.Title}\n\nReview: {review.Content}";

    var line = new
    {
        custom_id = $"review-{review.Index}",
        method    = "POST",
        url       = "/v1/chat/completions",
        body      = new
        {
            model    = modelDeployment,
            messages = new[]
            {
                new { role = "system", content = SystemPrompt },
                new { role = "user",   content = userContent  }
            },
            max_completion_tokens = 500, // o4-mini uses reasoning tokens internally before output
            response_format = new { type = "json_object" }
        }
    };

    jsonlBuilder.AppendLine(JsonSerializer.Serialize(line));
}

string jsonlContent  = jsonlBuilder.ToString();
long   fileSizeKb    = Encoding.UTF8.GetByteCount(jsonlContent) / 1024;
Console.WriteLine($"  Built {reviews.Count:N0} request lines ({fileSizeKb:N0} KB).");

// ── 5. Upload the .jsonl file ─────────────────────────────
Console.WriteLine();
Console.WriteLine("Step 3/8 — Uploading input file to Azure OpenAI...");

// The SDK's MIME type detection sends application/octet-stream for .jsonl,
// which the API rejects. Upload manually with application/json content type.
string fileId = await UploadBatchFileAsync(endpoint, apiKey, jsonlContent, http);

Console.WriteLine($"  Uploaded — file id: {fileId}");

// ── 6. Submit the batch job ───────────────────────────────
Console.WriteLine();
Console.WriteLine("Step 4/8 — Submitting batch job...");

string batchId = await CreateBatchAsync(endpoint, apiKey, fileId, http);

Console.WriteLine($"  Submitted — batch id: {batchId}");
Console.WriteLine();
Console.WriteLine("  Tip: if you want to stop and resume later, save this batch id.");
Console.WriteLine($"  Resume with: GET {endpoint}batches/{batchId}");

// ── 7. Poll until complete ────────────────────────────────
Console.WriteLine();
Console.WriteLine("Step 5/8 — Polling for completion (every 15s)...");
Console.WriteLine("  This typically takes a few minutes for 5,000 requests.");

var sw = Stopwatch.StartNew();
string currentStatus = "";

while (true)
{
    await Task.Delay(TimeSpan.FromSeconds(15));

    string pollJson = await GetBatchStatusAsync(endpoint, apiKey, batchId, http);
    using var pollDoc = JsonDocument.Parse(pollJson);
    currentStatus = pollDoc.RootElement.GetProperty("status").GetString() ?? "unknown";

    int? completed = pollDoc.RootElement.TryGetProperty("request_counts", out var counts)
        && counts.TryGetProperty("completed", out var c) ? c.GetInt32() : null;

    string progress = completed.HasValue ? $"{completed}/{ReviewCount}" : "...";
    Console.Write($"\r  [{sw.Elapsed:mm\\:ss}] status: {currentStatus,-12} requests: {progress}   ");

    if (currentStatus is "completed" or "failed" or "expired" or "cancelled")
        break;
}

sw.Stop();
Console.WriteLine();

// ── 8. Get final status ───────────────────────────────────
string finalJson = await GetBatchStatusAsync(endpoint, apiKey, batchId, http);
using var finalDoc = JsonDocument.Parse(finalJson);

string finalStatus = finalDoc.RootElement.GetProperty("status").GetString() ?? "unknown";
Console.WriteLine();
Console.WriteLine($"Step 6/8 — Batch finished in {sw.Elapsed.TotalMinutes:F1} min  (status: {finalStatus})");

bool hasOutput = finalDoc.RootElement.TryGetProperty("output_file_id", out var outputProp)
    && outputProp.ValueKind == JsonValueKind.String;

if (!hasOutput)
{
    bool hasError = finalDoc.RootElement.TryGetProperty("error_file_id", out var errProp)
        && errProp.ValueKind == JsonValueKind.String;
    Console.WriteLine("  Batch did not produce an output file.");
    if (hasError)
        Console.WriteLine($"  Error file id: {errProp.GetString()}");
    return;
}

// ── 9. Download & parse output ────────────────────────────
Console.WriteLine();
Console.WriteLine("Step 7/8 — Downloading and parsing results...");

string outputFileId = outputProp.GetString()!;
string outputJsonl  = await DownloadFileAsync(endpoint, apiKey, outputFileId, http);

// Build a lookup: custom_id → review for joining results
var reviewLookup = reviews.ToDictionary(r => $"review-{r.Index}");

var results = new List<SentimentResult>();
int parseErrors = 0;

foreach (string line in outputJsonl.Split('\n', StringSplitOptions.RemoveEmptyEntries))
{
    using var lineDoc = JsonDocument.Parse(line);
    var root = lineDoc.RootElement;

    string customId = root.GetProperty("custom_id").GetString() ?? "";

    // Check for request-level error
    if (root.TryGetProperty("error", out var errEl) && errEl.ValueKind != JsonValueKind.Null)
    {
        parseErrors++;
        continue;
    }

    // Extract the model's text content
    string? content = root
        .GetProperty("response")
        .GetProperty("body")
        .GetProperty("choices")[0]
        .GetProperty("message")
        .GetProperty("content")
        .GetString();

    if (content is null || !reviewLookup.TryGetValue(customId, out var review))
    {
        parseErrors++;
        continue;
    }

    // Parse the model's JSON output
    try
    {
        using var modelJson = JsonDocument.Parse(content);
        var m = modelJson.RootElement;

        results.Add(new SentimentResult(
            CustomId:        customId,
            ActualLabel:     review.ActualLabel,
            Title:           review.Title,
            ContentExcerpt:  review.Content.Length > 80 ? review.Content[..77] + "..." : review.Content,
            Sentiment:       m.TryGetProperty("sentiment",     out var s) ? s.GetString() ?? "" : "",
            HasComplaint:    m.TryGetProperty("has_complaint", out var h) && h.GetBoolean(),
            CategoryTag:     m.TryGetProperty("category_tag", out var t) ? t.GetString() ?? "" : ""
        ));
    }
    catch
    {
        parseErrors++;
    }
}

Console.WriteLine($"  Parsed {results.Count:N0} results  ({parseErrors} errors)");

// ── 10. Save to CSV ───────────────────────────────────────
Console.WriteLine();
Console.WriteLine("Step 8/8 — Saving results to CSV...");

string csvPath = Path.Combine(AppContext.BaseDirectory, "batch_results.csv");

using var writer = new StreamWriter(csvPath, append: false, Encoding.UTF8);
await writer.WriteLineAsync("custom_id,actual_label,predicted_sentiment,has_complaint,category_tag,review_excerpt");

foreach (var r in results)
{
    string excerpt = r.ContentExcerpt.Replace("\"", "\"\""); // escape quotes for CSV
    await writer.WriteLineAsync(
        $"{r.CustomId},{(r.ActualLabel == 1 ? "positive" : "negative")},{r.Sentiment},{r.HasComplaint},{r.CategoryTag},\"{excerpt}\""
    );
}

Console.WriteLine($"  Saved to: {csvPath}");

// ── 11. Accuracy report ───────────────────────────────────
Console.WriteLine();
Console.WriteLine("══════════════════════════════════════════════");
Console.WriteLine("  Accuracy Report");
Console.WriteLine("══════════════════════════════════════════════");

// Only compare where model returned positive or negative (exclude neutral)
var comparable = results.Where(r => r.Sentiment is "positive" or "negative").ToList();

int correct = comparable.Count(r =>
    (r.ActualLabel == 1 && r.Sentiment == "positive") ||
    (r.ActualLabel == 0 && r.Sentiment == "negative")
);

Console.WriteLine($"  Total results     : {results.Count:N0}");
Console.WriteLine($"  Parse errors      : {parseErrors}");
Console.WriteLine($"  Neutral (skipped) : {results.Count(r => r.Sentiment == "neutral")}");
Console.WriteLine($"  Comparable        : {comparable.Count:N0}");
Console.WriteLine($"  Correct           : {correct:N0}");
Console.WriteLine($"  Accuracy          : {(double)correct / comparable.Count:P1}");
Console.WriteLine();

// Category breakdown
Console.WriteLine("  Category breakdown:");
Console.WriteLine($"  {"Category",-12}  {"Count",6}  {"Complaints",10}");
Console.WriteLine($"  {new string('-', 12)}  {new string('-', 6)}  {new string('-', 10)}");

foreach (var grp in results.GroupBy(r => r.CategoryTag).OrderByDescending(g => g.Count()))
{
    int complaints = grp.Count(r => r.HasComplaint);
    Console.WriteLine($"  {grp.Key,-12}  {grp.Count(),6}  {complaints,10}");
}

Console.WriteLine();
Console.WriteLine("Done!");

// ── Raw API helpers ───────────────────────────────────────
//
//  We bypass the OpenAI SDK for file/batch operations because
//  this Foundry endpoint uses the OpenAI-compatible /v1/ API
//  and the SDK's content type detection causes upload failures.

static async Task<string> UploadBatchFileAsync(
    string endpoint, string apiKey, string jsonlContent, HttpClient http)
{
    using var multipart = new MultipartFormDataContent();
    multipart.Add(new StringContent("batch"), "purpose");

    var fileContent = new ByteArrayContent(Encoding.UTF8.GetBytes(jsonlContent));
    fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
    multipart.Add(fileContent, "file", "batch_input.jsonl");

    using var request = new HttpRequestMessage(HttpMethod.Post, $"{endpoint}files");
    request.Headers.Add("api-key", apiKey);
    request.Content = multipart;

    var response = await http.SendAsync(request);
    response.EnsureSuccessStatusCode();

    using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
    return doc.RootElement.GetProperty("id").GetString()
        ?? throw new InvalidOperationException("No file id in upload response");
}

static async Task<string> CreateBatchAsync(
    string endpoint, string apiKey, string inputFileId, HttpClient http)
{
    var body = JsonSerializer.Serialize(new
    {
        input_file_id     = inputFileId,
        endpoint          = "/v1/chat/completions",
        completion_window = "24h"
    });

    using var request = new HttpRequestMessage(HttpMethod.Post, $"{endpoint}batches");
    request.Headers.Add("api-key", apiKey);
    request.Content = new StringContent(body, Encoding.UTF8, "application/json");

    var response = await http.SendAsync(request);
    string responseBody = await response.Content.ReadAsStringAsync();
    if (!response.IsSuccessStatusCode)
        throw new InvalidOperationException($"Batch creation failed ({response.StatusCode}): {responseBody}");

    using var doc = JsonDocument.Parse(responseBody);
    return doc.RootElement.GetProperty("id").GetString()
        ?? throw new InvalidOperationException("No batch id in response");
}

static async Task<string> GetBatchStatusAsync(
    string endpoint, string apiKey, string batchId, HttpClient http)
{
    using var request = new HttpRequestMessage(HttpMethod.Get, $"{endpoint}batches/{batchId}");
    request.Headers.Add("api-key", apiKey);

    var response = await http.SendAsync(request);
    response.EnsureSuccessStatusCode();
    return await response.Content.ReadAsStringAsync();
}

static async Task<string> DownloadFileAsync(
    string endpoint, string apiKey, string fileId, HttpClient http)
{
    using var request = new HttpRequestMessage(HttpMethod.Get, $"{endpoint}files/{fileId}/content");
    request.Headers.Add("api-key", apiKey);

    var response = await http.SendAsync(request);
    response.EnsureSuccessStatusCode();
    return await response.Content.ReadAsStringAsync();
}

// ── Result record ─────────────────────────────────────────
record SentimentResult(
    string CustomId,
    int    ActualLabel,
    string Title,
    string ContentExcerpt,
    string Sentiment,
    bool   HasComplaint,
    string CategoryTag
);
