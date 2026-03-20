using Microsoft.Extensions.AI;

/// <summary>
/// In-memory semantic cache that uses embeddings + cosine similarity to find
/// cached responses for queries that are semantically similar (not just exact matches).
/// </summary>
public class SemanticCache
{
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;
    private readonly List<(ReadOnlyMemory<float> Embedding, string Response)> _entries = new();
    private readonly float _threshold;

    /// <param name="embeddingGenerator">Used to embed queries into vectors.</param>
    /// <param name="threshold">
    ///   Cosine similarity threshold (0–1). Higher = stricter matching.
    ///   0.95 catches near-identical phrasings; 0.85 catches broader paraphrases.
    /// </param>
    public SemanticCache(
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
        float threshold = 0.95f)
    {
        _embeddingGenerator = embeddingGenerator;
        _threshold = threshold;
    }

    /// <summary>
    /// Embeds the query and searches for a semantically similar cached response.
    /// Returns null on cache miss.
    /// </summary>
    public async Task<string?> FindAsync(string query, CancellationToken ct = default)
    {
        if (_entries.Count == 0) return null;

        var results = await _embeddingGenerator.GenerateAsync([query], cancellationToken: ct);
        var queryVector = results[0].Vector;

        foreach (var (embedding, response) in _entries)
        {
            float similarity = CosineSimilarity(queryVector.Span, embedding.Span);
            Console.WriteLine($"  [SemanticCache] Similarity: {similarity:F4} (threshold: {_threshold})");

            if (similarity >= _threshold)
                return response;
        }

        return null;
    }

    /// <summary>
    /// Embeds the query and stores the response in the cache.
    /// </summary>
    public async Task StoreAsync(string query, string response, CancellationToken ct = default)
    {
        var results = await _embeddingGenerator.GenerateAsync([query], cancellationToken: ct);
        _entries.Add((results[0].Vector, response));
    }

    /// <summary>
    /// Computes cosine similarity between two float vectors.
    /// Returns a value between -1 (opposite) and 1 (identical).
    /// </summary>
    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = 0f, magA = 0f, magB = 0f;

        for (int i = 0; i < a.Length; i++)
        {
            dot  += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }

        return dot / (MathF.Sqrt(magA) * MathF.Sqrt(magB));
    }
}
