// ============================================================
//  MAF Learning – 12: Basic Tool Calling
//
//  Demonstrates how to define tools with AIFunctionFactory
//  and let the agent invoke them automatically via
//  UseFunctionInvocation() middleware.
//
//  Tools defined here:
//    - get_current_time  : returns current UTC timestamp
//    - get_weather       : returns mock weather for a city
//    - add_numbers       : adds two numbers (shows typed args)
// ============================================================

using Azure;
using Azure.AI.OpenAI;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
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
Console.WriteLine("  MAF Learning – 12: Basic Tool Calling");
Console.WriteLine("==============================================");
Console.WriteLine($"  Endpoint : {endpoint}");
Console.WriteLine($"  Model    : {modelDeployment}");
Console.WriteLine();

// ── 2. Define tools ───────────────────────────────────────
//
//  AIFunctionFactory.Create() wraps any delegate as a tool.
//  The delegate's parameter names and types become the tool's
//  JSON schema — the model fills them in from the user's query.
//
//  UseFunctionInvocation() (added in step 4) automatically:
//    a) detects when the model emits a tool-call
//    b) invokes the matching delegate
//    c) feeds the result back to the model
//    d) repeats until the model returns a final text response

AIFunction getCurrentTime = AIFunctionFactory.Create(
    () =>
    {
        Console.WriteLine("  → [tool] get_current_time()");
        return DateTime.UtcNow.ToString("R");
    },
    name: "get_current_time",
    description: "Returns the current UTC date and time."
);

AIFunction getWeather = AIFunctionFactory.Create(
    (string city) =>
    {
        Console.WriteLine($"  → [tool] get_weather(city: \"{city}\")");
        // Simulated weather data — replace with a real API call as needed
        var mock = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["London"]   = "12°C, overcast",
            ["New York"]  = "18°C, sunny",
            ["Tokyo"]    = "22°C, partly cloudy",
            ["Sydney"]   = "25°C, clear skies",
        };
        return mock.TryGetValue(city, out string? weather)
            ? $"Weather in {city}: {weather}"
            : $"No weather data available for '{city}'.";
    },
    name: "get_weather",
    description: "Returns the current weather for a given city. Supported cities: London, New York, Tokyo, Sydney."
);

AIFunction addNumbers = AIFunctionFactory.Create(
    (double a, double b) =>
    {
        Console.WriteLine($"  → [tool] add_numbers(a: {a}, b: {b})");
        return a + b;
    },
    name: "add_numbers",
    description: "Adds two numbers together and returns the result."
);

// ── 3. Build Azure OpenAI client ──────────────────────────
AzureOpenAIClient azureClient = new(
    new Uri(endpoint),
    new AzureKeyCredential(apiKey),
    new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_12_01_Preview)
);

// ── 4. Build IChatClient with UseFunctionInvocation() ─────
//
//  UseFunctionInvocation() is the key middleware that enables
//  tool calling. Without it, the model's tool-call messages
//  would be returned as-is with no invocation.
IChatClient chatClient = azureClient
    .GetChatClient(modelDeployment)
    .AsIChatClient()
    .AsBuilder()
    .UseFunctionInvocation()
    .Build();

// ── 5. Create agent with all three tools ──────────────────
ChatClientAgent agent = new ChatClientAgent(
    chatClient: chatClient,
    name: "ToolCallingAgent",
    instructions: "You are a helpful assistant. Use the available tools whenever the user's question requires it.",
    tools: [getCurrentTime, getWeather, addNumbers]
);

AgentSession session = await agent.CreateSessionAsync();

// ── 6. Run queries that exercise each tool ────────────────
await AskAsync("What time is it right now?");
await AskAsync("What's the weather like in Tokyo?");
await AskAsync("What is 1337 + 42?");
await AskAsync("What is the weather in London and what time is it?");

Console.WriteLine("Done!");

// ── Helper ────────────────────────────────────────────────
async Task AskAsync(string question)
{
    Console.WriteLine($"User  : {question}");
    AgentResponse response = await agent.RunAsync(message: question, session: session);
    Console.WriteLine($"Agent : {response.Text}");
    Console.WriteLine();
}
