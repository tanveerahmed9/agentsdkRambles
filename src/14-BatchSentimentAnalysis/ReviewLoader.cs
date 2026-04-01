// ============================================================
//  ReviewLoader
//
//  Downloads Amazon product reviews from the HuggingFace
//  Datasets Server API (no auth required).
//
//  Dataset: fancyzhx/amazon_polarity
//    label   — 0 = negative, 1 = positive  (ground truth)
//    title   — short review headline
//    content — full review text
//
//  We paginate in batches of 100 (API maximum per request)
//  until we have the requested number of reviews.
// ============================================================

using System.Text.Json;
using System.Text.Json.Serialization;

public record AmazonReview(
    int    Index,        // row index in the dataset (used as custom_id)
    int    ActualLabel,  // 0 = negative, 1 = positive (ground truth for validation)
    string Title,
    string Content
);

public static class ReviewLoader
{
    private const string BaseUrl   = "https://datasets-server.huggingface.co/rows";
    private const string Dataset   = "fancyzhx/amazon_polarity";
    private const string Config    = "amazon_polarity";
    private const string Split     = "test";
    private const int    PageSize        = 100;  // HuggingFace API maximum
    private const int    MaxReviewLength = 600;  // chars — caps token cost per review
    private const int    PageDelayMs     = 500;  // delay between pages to avoid rate limits
    private static readonly string CachePath =
        Path.Combine(Path.GetTempPath(), "maf_amazon_reviews_cache.json");

    public static async Task<List<AmazonReview>> LoadAsync(int count, HttpClient http)
    {
        // Return from disk cache if available — avoids re-downloading on retry runs
        if (File.Exists(CachePath))
        {
            var cached = JsonSerializer.Deserialize<List<AmazonReview>>(
                await File.ReadAllTextAsync(CachePath));
            if (cached != null && cached.Count >= count)
            {
                Console.WriteLine($"  Loaded {count} reviews from cache ({CachePath})");
                return cached.Take(count).ToList();
            }
        }

        var reviews = new List<AmazonReview>(count);
        int pages   = (int)Math.Ceiling(count / (double)PageSize);

        Console.WriteLine($"  Fetching {count} reviews ({pages} pages × {PageSize} rows)...");

        for (int page = 0; page < pages; page++)
        {
            int offset = page * PageSize;
            int length = Math.Min(PageSize, count - offset);
            string url = $"{BaseUrl}?dataset={Dataset}&config={Config}&split={Split}&offset={offset}&length={length}";

            HttpResponseMessage response = await GetWithRetryAsync(http, url);
            string json = await response.Content.ReadAsStringAsync();

            using var doc = JsonDocument.Parse(json);
            int pageStart = reviews.Count;

            foreach (var row in doc.RootElement.GetProperty("rows").EnumerateArray())
            {
                var r      = row.GetProperty("row");
                int label  = r.GetProperty("label").GetInt32();
                string title   = r.GetProperty("title").GetString()   ?? "";
                string content = r.GetProperty("content").GetString() ?? "";

                if (content.Length > MaxReviewLength)
                    content = content[..MaxReviewLength] + "…";

                reviews.Add(new AmazonReview(
                    Index:       offset + (reviews.Count - pageStart),
                    ActualLabel: label,
                    Title:       title,
                    Content:     content
                ));
            }

            Console.Write($"\r  Fetched {reviews.Count}/{count} reviews...");
            await Task.Delay(PageDelayMs);
        }

        Console.WriteLine();

        // Save to cache for future runs
        await File.WriteAllTextAsync(CachePath, JsonSerializer.Serialize(reviews));
        Console.WriteLine($"  Cached to {CachePath}");

        return reviews;
    }

    private static async Task<HttpResponseMessage> GetWithRetryAsync(HttpClient http, string url)
    {
        int[] backoffSeconds = [3, 10, 30, 60, 120];

        for (int attempt = 0; ; attempt++)
        {
            var response = await http.GetAsync(url);

            if (response.IsSuccessStatusCode)
                return response;

            if ((int)response.StatusCode == 429 && attempt < backoffSeconds.Length)
            {
                int wait = backoffSeconds[attempt];
                Console.Write($"\r  Rate limited — waiting {wait}s...                ");
                await Task.Delay(TimeSpan.FromSeconds(wait));
                continue;
            }

            response.EnsureSuccessStatusCode(); // throws for non-429 errors
            return response;
        }
    }
}
