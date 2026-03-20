using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using Microsoft.Extensions.AI;
using StackExchange.Redis;

/// <summary>
/// Hybrid semantic cache backed by Azure AI Search (vector similarity lookup)
/// and Azure Cache for Redis (fast response storage).
///
/// Write path:
///   1. Embed the query with ada-002
///   2. Store response text in Redis (key = new GUID)
///   3. Index the query text + vector + Redis key in Azure AI Search
///
/// Read path:
///   1. Embed the incoming query
///   2. Run a vector search in Azure AI Search (nearest neighbour)
///   3. If top result score >= threshold → fetch response from Redis by key
///   4. Otherwise → cache miss
/// </summary>
public class HybridCache
{
    private readonly SearchClient _searchClient;
    private readonly SearchIndexClient _indexClient;
    private readonly IDatabase _redis;
    private readonly IEmbeddingGenerator<string, Embedding<float>> _embeddingGenerator;
    private readonly string _indexName;
    private readonly float _threshold;

    private const string VectorProfileName = "hnsw-profile";
    private const string VectorAlgorithmName = "hnsw-config";

    // Dimensions vary by model:
    //   text-embedding-ada-002  → 1536
    //   text-embedding-3-large  → 3072
    //   sentence-transformers   → typically 768
    private readonly int _embeddingDimensions;

    public HybridCache(
        SearchClient searchClient,
        SearchIndexClient indexClient,
        IDatabase redis,
        IEmbeddingGenerator<string, Embedding<float>> embeddingGenerator,
        string indexName,
        float threshold = 0.88f,
        int embeddingDimensions = 1536)
    {
        _searchClient = searchClient;
        _indexClient = indexClient;
        _redis = redis;
        _embeddingGenerator = embeddingGenerator;
        _indexName = indexName;
        _threshold = threshold;
        _embeddingDimensions = embeddingDimensions;
    }

    // ── Index management ──────────────────────────────────────────────────────

    /// <summary>
    /// Creates the Azure AI Search index if it does not already exist.
    /// Defines a vector field (HNSW, cosine similarity) for semantic lookup.
    /// </summary>
    public async Task EnsureIndexExistsAsync(CancellationToken ct = default)
    {
        try
        {
            await _indexClient.GetIndexAsync(_indexName, ct);
            Console.WriteLine($"  [AzureSearch] Index '{_indexName}' already exists.");
        }
        catch (RequestFailedException ex) when (ex.Status == 404)
        {
            Console.WriteLine($"  [AzureSearch] Creating index '{_indexName}'...");

            var index = new SearchIndex(_indexName)
            {
                Fields =
                {
                    new SimpleField("id", SearchFieldDataType.String)
                        { IsKey = true, IsFilterable = true },

                    new SearchableField("queryText")
                        { IsFilterable = false },

                    new SimpleField("responseId", SearchFieldDataType.String),

                    new VectorSearchField("queryVector", _embeddingDimensions, VectorProfileName)
                },
                VectorSearch = new VectorSearch
                {
                    Profiles  = { new VectorSearchProfile(VectorProfileName, VectorAlgorithmName) },
                    Algorithms =
                    {
                        new HnswAlgorithmConfiguration(VectorAlgorithmName)
                        {
                            Parameters = new HnswParameters
                            {
                                Metric = VectorSearchAlgorithmMetric.Cosine
                            }
                        }
                    }
                }
            };

            await _indexClient.CreateIndexAsync(index, ct);
            Console.WriteLine($"  [AzureSearch] Index '{_indexName}' created.");
        }
    }

    // ── Read path ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Embeds the query, searches AI Search for a similar cached query,
    /// and fetches the response from Redis on a hit.
    /// Returns null on cache miss.
    /// </summary>
    public async Task<string?> FindAsync(string query, CancellationToken ct = default)
    {
        // 1. Embed the query
        var embeddings = await _embeddingGenerator.GenerateAsync([query], cancellationToken: ct);
        float[] queryVector = embeddings[0].Vector.ToArray();

        // 2. Vector search in Azure AI Search — find the nearest cached query
        var vectorQuery = new VectorizedQuery(queryVector)
        {
            KNearestNeighborsCount = 1,
            Fields = { "queryVector" }
        };

        var searchOptions = new SearchOptions
        {
            VectorSearch = new VectorSearchOptions { Queries = { vectorQuery } },
            Select = { "id", "responseId", "queryText" },
            Size = 1
        };

        SearchResults<CacheDocument> results =
            await _searchClient.SearchAsync<CacheDocument>(searchText: null, searchOptions, ct);

        await foreach (SearchResult<CacheDocument> result in results.GetResultsAsync())
        {
            double score = result.Score ?? 0;
            Console.WriteLine($"  [AzureSearch] Nearest match score: {score:F4} (threshold: {_threshold})");

            if (score >= _threshold)
            {
                // 3. Fetch cached response from Redis
                string responseId = result.Document.ResponseId;
                RedisValue cached = await _redis.StringGetAsync(responseId);

                if (cached.HasValue)
                {
                    Console.WriteLine($"  [Redis] HIT — key: {responseId}");
                    return cached.ToString();
                }

                Console.WriteLine($"  [Redis] Key {responseId} missing — treating as cache miss.");
            }
        }

        return null;
    }

    // ── Write path ────────────────────────────────────────────────────────────

    /// <summary>
    /// Embeds the query, stores the response in Redis,
    /// then indexes the query + vector + Redis key in Azure AI Search.
    /// </summary>
    public async Task StoreAsync(string query, string response, CancellationToken ct = default)
    {
        string responseId = Guid.NewGuid().ToString();

        // 1. Store response text in Redis (no expiry — adjust as needed)
        await _redis.StringSetAsync(responseId, response);
        Console.WriteLine($"  [Redis] Stored response — key: {responseId}");

        // 2. Embed the query
        var embeddings = await _embeddingGenerator.GenerateAsync([query], cancellationToken: ct);
        float[] queryVector = embeddings[0].Vector.ToArray();

        // 3. Index query + vector + Redis key in Azure AI Search
        var document = new CacheDocument
        {
            Id         = responseId,
            QueryText  = query,
            QueryVector = queryVector,
            ResponseId = responseId
        };

        await _searchClient.MergeOrUploadDocumentsAsync([document], cancellationToken: ct);
        Console.WriteLine($"  [AzureSearch] Indexed query for future similarity lookups.");
    }
}
