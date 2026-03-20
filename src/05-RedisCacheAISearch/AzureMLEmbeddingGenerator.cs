using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;

/// <summary>
/// Custom IEmbeddingGenerator that calls an Azure ML managed online endpoint
/// hosting a domain-specific embedding model (e.g. a sentence-transformer
/// fine-tuned on energy domain data).
///
/// Expected Azure ML endpoint request format:
///   POST {endpoint}/score
///   Headers: Authorization: Bearer {apiKey}
///   Body: { "input_data": ["text1", "text2"] }
///
/// Expected response format:
///   [[0.1, 0.2, ...], [0.3, 0.4, ...]]
///
/// Note: If your Azure ML model uses a different input/output schema,
/// adjust the request/response models below accordingly.
/// </summary>
public class AzureMLEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly HttpClient _httpClient;
    private readonly string _endpoint;
    private readonly int _dimensions;

    public AzureMLEmbeddingGenerator(string endpoint, string apiKey, int dimensions)
    {
        _endpoint = endpoint;
        _dimensions = dimensions;

        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
    }

    public EmbeddingGeneratorMetadata Metadata =>
        new("AzureML", new Uri(_endpoint));

    /// <summary>
    /// Sends texts to the Azure ML endpoint and returns embeddings.
    /// </summary>
    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var inputs = values.ToList();

        // ── Request ────────────────────────────────────────────────────────
        //  Azure ML managed online endpoints for sentence-transformers
        //  typically accept this format. Adjust if your model differs.
        var requestBody = new AzureMLEmbeddingRequest { InputData = inputs };

        var response = await _httpClient.PostAsJsonAsync(
            _endpoint,
            requestBody,
            cancellationToken
        );

        response.EnsureSuccessStatusCode();

        // ── Response ───────────────────────────────────────────────────────
        //  Response is a 2D array: [[float, float, ...], [float, float, ...]]
        var vectors = await response.Content
            .ReadFromJsonAsync<List<List<float>>>(cancellationToken: cancellationToken)
            ?? throw new InvalidOperationException("Azure ML returned null embedding response.");

        var embeddings = vectors
            .Select(v => new Embedding<float>(v.ToArray()))
            .ToList();

        Console.WriteLine($"  [AzureML] Generated {embeddings.Count} embedding(s) " +
                          $"(dims: {embeddings[0].Vector.Length})");

        return new GeneratedEmbeddings<Embedding<float>>(embeddings);
    }

    public object? GetService(Type serviceType, object? key = null) => null;

    public void Dispose() => _httpClient.Dispose();
}

// ── Request / response models ─────────────────────────────────────────────────

internal class AzureMLEmbeddingRequest
{
    [JsonPropertyName("input_data")]
    public List<string> InputData { get; set; } = [];
}
