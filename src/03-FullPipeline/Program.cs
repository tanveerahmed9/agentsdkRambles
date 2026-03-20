// ============================================================
//  MAF Learning – 03: Full Middleware Pipeline
//
//  Middleware chain (outermost → innermost):
//    1. UseLogging()           → logs every request/response to console
//    2. UseOpenTelemetry()     → traces + metrics to Application Insights
//    3. UseFunctionInvocation()→ auto-handles tool calls (placeholder tools for now)
//    4. UseDistributedCache()  → caches responses in-memory
// ============================================================

using Azure;
using Azure.AI.OpenAI;
using Azure.Monitor.OpenTelemetry.Exporter;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using OpenTelemetry;
using OpenTelemetry.Metrics;
using OpenTelemetry.Trace;

// ── 1. Load .env ──────────────────────────────────────────────
DotNetEnv.Env.TraversePath().Load();

// ── 2. Load config ────────────────────────────────────────────
IConfiguration config = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

string endpoint        = config["AzureAI:Endpoint"]              ?? throw new InvalidOperationException("AzureAI:Endpoint missing");
string apiKey          = config["AzureAI:ApiKey"]                ?? throw new InvalidOperationException("AzureAI:ApiKey missing");
string modelDeployment = config["AzureAI:ModelDeployment"]       ?? "o4-mini";
string appInsightsCs   = config["AzureMonitor:ConnectionString"] ?? throw new InvalidOperationException("AzureMonitor:ConnectionString missing");

if (!appInsightsCs.Contains("InstrumentationKey=", StringComparison.OrdinalIgnoreCase))
    throw new InvalidOperationException("AzureMonitor:ConnectionString is invalid — must contain 'InstrumentationKey='");

Console.WriteLine("==============================================");
Console.WriteLine("  MAF Learning – 03: Full Middleware Pipeline");
Console.WriteLine("==============================================");
Console.WriteLine($"  Endpoint : {endpoint}");
Console.WriteLine($"  Model    : {modelDeployment}");
Console.WriteLine();

// ── 3. OpenTelemetry → Application Insights ──────────────────
using TracerProvider tracerProvider = Sdk.CreateTracerProviderBuilder()
    .AddSource("Microsoft.Extensions.AI")
    .AddAzureMonitorTraceExporter(o => o.ConnectionString = appInsightsCs)
    .Build();

using MeterProvider meterProvider = Sdk.CreateMeterProviderBuilder()
    .AddMeter("Microsoft.Extensions.AI")
    .AddAzureMonitorMetricExporter(o => o.ConnectionString = appInsightsCs)
    .Build();

// ── 4. Console logger ─────────────────────────────────────────
using ILoggerFactory loggerFactory = LoggerFactory.Create(b => b
    .AddConsole()
    .SetMinimumLevel(LogLevel.Debug)
);

// ── 5. In-memory distributed cache ───────────────────────────
IDistributedCache cache = new MemoryDistributedCache(
    Options.Create(new MemoryDistributedCacheOptions())
);

// ── 6. Placeholder tools (to be replaced with real tools later)
//  AIFunctionFactory.Create() wraps any delegate as a callable tool.
//  UseFunctionInvocation() will auto-invoke these when the model calls them.
AIFunction getCurrentTime = AIFunctionFactory.Create(
    () => DateTime.UtcNow.ToString("R"),
    name: "get_current_time",
    description: "Returns the current UTC date and time."
);

// ── 7. Build IChatClient with full middleware pipeline ────────
//
//  Request flow (top → bottom):
//    UseLogging          → logs raw messages to console
//    UseOpenTelemetry    → emits spans + metrics
//    UseFunctionInvocation → intercepts tool calls, invokes them, loops back
//    UseDistributedCache → returns cached response if same prompt seen before
//    Azure OpenAI        → actual model call (only reached on cache miss)
//
AzureOpenAIClient azureClient = new(
    new Uri(endpoint),
    new AzureKeyCredential(apiKey),
    new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_12_01_Preview)
);

IChatClient chatClient = azureClient
    .GetChatClient(modelDeployment)
    .AsIChatClient()
    .AsBuilder()
    .UseLogging(loggerFactory)
    .UseOpenTelemetry(
        sourceName: "Microsoft.Extensions.AI",
        configure: o => o.EnableSensitiveData = false
    )
    .UseFunctionInvocation()
    .UseDistributedCache(cache)
    .Build();

// ── 8. Create agent and session ───────────────────────────────
ChatClientAgent agent = new ChatClientAgent(
    chatClient: chatClient,
    name: "FullPipelineAgent",
    instructions: "You are a helpful AI assistant. Answer clearly and concisely.",
    tools: [getCurrentTime]
);

AgentSession session = await agent.CreateSessionAsync();

// ── 9. First query (cache miss — hits the model) ──────────────
string userQuery = "What are the benefits of middleware in software architecture?";

Console.WriteLine($"User  : {userQuery}");
Console.WriteLine();
Console.WriteLine("Agent [1st call — cache miss]:");

AgentResponse response = await agent.RunAsync(message: userQuery, session: session);
Console.WriteLine(response.Text);
Console.WriteLine();

// ── 10. Same query again (cache hit — model not called) ───────
Console.WriteLine($"User  : {userQuery} (same query)");
Console.WriteLine();
Console.WriteLine("Agent [2nd call — cache hit]:");

AgentResponse cachedResponse = await agent.RunAsync(message: "what is the capital of France?", session: session);
Console.WriteLine(cachedResponse.Text);
Console.WriteLine();

// ── 11. Flush telemetry ───────────────────────────────────────
bool tracesExported  = tracerProvider.ForceFlush(10_000);
bool metricsExported = meterProvider.ForceFlush(10_000);

if (!tracesExported || !metricsExported)
    throw new InvalidOperationException("Telemetry export to Application Insights failed.");

Console.WriteLine($"  Traces  exported : {(tracesExported  ? "OK" : "FAILED")}");
Console.WriteLine($"  Metrics exported : {(metricsExported ? "OK" : "FAILED")}");
Console.WriteLine("Done!");
