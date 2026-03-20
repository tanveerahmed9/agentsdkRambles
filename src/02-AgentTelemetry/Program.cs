// ============================================================
//  MAF Learning – 02: Agent Telemetry with Application Insights
//
//  Packages:
//    Azure.AI.OpenAI 2.8.0-beta.1               → AzureOpenAIClient
//    Microsoft.Extensions.AI.OpenAI 10.3.0      → IChatClient / UseOpenTelemetry()
//    Microsoft.Agents.AI 1.0.0-rc3              → ChatClientAgent / AgentSession
//    OpenTelemetry 1.9.0                        → TracerProvider / MeterProvider
//    Azure.Monitor.OpenTelemetry.Exporter 1.3.0 → AddAzureMonitorTraceExporter()
//
//  Environment variables required:
//    AzureAI__Endpoint
//    AzureAI__ApiKey
//    AzureMonitor__ConnectionString
// ============================================================

using Azure;
using Azure.AI.OpenAI;
using Azure.Monitor.OpenTelemetry.Exporter;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using OpenTelemetry;
using OpenTelemetry.Trace;
using OpenTelemetry.Metrics;

// ── 1. Load config ────────────────────────────────────────────
IConfiguration config = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

string endpoint        = config["AzureAI:Endpoint"]        ?? throw new InvalidOperationException("AzureAI:Endpoint missing");
string apiKey          = config["AzureAI:ApiKey"]          ?? throw new InvalidOperationException("AzureAI:ApiKey missing");
string modelDeployment = config["AzureAI:ModelDeployment"] ?? "o4-mini";
string appInsightsCs   = config["AzureMonitor:ConnectionString"] ?? throw new InvalidOperationException("AzureMonitor:ConnectionString missing");

// Validate connection string format before attempting to use it
if (!appInsightsCs.Contains("InstrumentationKey=", StringComparison.OrdinalIgnoreCase))
    throw new InvalidOperationException("AzureMonitor:ConnectionString is invalid — must contain 'InstrumentationKey='");

Console.WriteLine("==============================================");
Console.WriteLine("  MAF Learning – 02: Agent Telemetry");
Console.WriteLine("==============================================");
Console.WriteLine($"  Endpoint : {endpoint}");
Console.WriteLine($"  Model    : {modelDeployment}");
Console.WriteLine($"  AppInsights connected: true");
Console.WriteLine();

// ── 2. Set up OpenTelemetry → Application Insights ───────────
//  TracerProvider captures spans from the IChatClient middleware
//  and exports them to App Insights via Azure Monitor exporter.
//  MeterProvider captures token usage / latency metrics.
using TracerProvider tracerProvider = Sdk.CreateTracerProviderBuilder()
    .AddSource("Microsoft.Extensions.AI")   // emitted by UseOpenTelemetry() middleware
    .AddAzureMonitorTraceExporter(o => o.ConnectionString = appInsightsCs)
    .Build();

using MeterProvider meterProvider = Sdk.CreateMeterProviderBuilder()
    .AddMeter("Microsoft.Extensions.AI")    // token usage, latency histograms
    .AddAzureMonitorMetricExporter(o => o.ConnectionString = appInsightsCs)
    .Build();

// ── 3. Create Azure OpenAI client with telemetry middleware ───
//  .UseOpenTelemetry() wraps the IChatClient and emits:
//    - A span per chat completion (prompt, model, finish reason)
//    - Metrics: token counts, request duration
AzureOpenAIClient azureClient = new(
    new Uri(endpoint),
    new AzureKeyCredential(apiKey),
    new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_12_01_Preview)
);

IChatClient chatClient = azureClient
    .GetChatClient(modelDeployment)
    .AsIChatClient()
    .AsBuilder()
    .UseOpenTelemetry(
        sourceName: "Microsoft.Extensions.AI",
        configure: o => o.EnableSensitiveData = false   // set true locally to log prompt content
    )
    .Build();

// ── 4. Create agent and session ───────────────────────────────
ChatClientAgent agent = new ChatClientAgent(
    chatClient: chatClient,
    name: "TelemetryAgent",
    instructions: "You are a helpful AI assistant. Answer clearly and concisely."
);

AgentSession session = await agent.CreateSessionAsync();

// ── 5. Send a query ───────────────────────────────────────────
string userQuery = "What is OpenTelemetry and why is it useful for AI agents?";

Console.WriteLine($"User  : {userQuery}");
Console.WriteLine();
Console.WriteLine("Agent : ");

AgentResponse response = await agent.RunAsync(
    message: userQuery,
    session: session
);

Console.WriteLine(response.Text);
Console.WriteLine();

// ── 6. Flush telemetry and verify export succeeded ────────────
//  ForceFlush sends all buffered spans/metrics immediately.
//  Returns false if the export failed (e.g. wrong connection string,
//  network issue, or App Insights endpoint unreachable).
bool tracesExported  = tracerProvider.ForceFlush(10_000);
bool metricsExported = meterProvider.ForceFlush(10_000);

if (!tracesExported || !metricsExported)
    throw new InvalidOperationException(
        "Telemetry export to Application Insights failed — check your connection string and network connectivity.");

Console.WriteLine($"  Traces  exported : {(tracesExported  ? "OK" : "FAILED")}");
Console.WriteLine($"  Metrics exported : {(metricsExported ? "OK" : "FAILED")}");
Console.WriteLine("Done! Traces and metrics sent to Application Insights.");
