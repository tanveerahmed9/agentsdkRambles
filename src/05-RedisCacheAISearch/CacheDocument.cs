using System.Text.Json.Serialization;

/// <summary>
/// Document stored in the Azure AI Search index.
/// Each entry represents one cached query.
///
///  id          → unique GUID, also used as the Redis key to fetch the response
///  queryText   → original query text (for debugging / inspection in the portal)
///  queryVector → ada-002 embedding of the query (1536 dims, used for vector search)
///  responseId  → Redis key where the cached response text is stored
/// </summary>
public class CacheDocument
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = string.Empty;

    [JsonPropertyName("queryText")]
    public string QueryText { get; set; } = string.Empty;

    [JsonPropertyName("queryVector")]
    public IReadOnlyList<float> QueryVector { get; set; } = [];

    [JsonPropertyName("responseId")]
    public string ResponseId { get; set; } = string.Empty;
}
