// ============================================================
//  MAF Learning – 01: Basic Agent Query
//
//  Packages:
//    Azure.AI.OpenAI 2.1.0              → AzureOpenAIClient
//    Microsoft.Extensions.AI.OpenAI     → IChatClient / AsChatClient()
//    Microsoft.Agents.AI 1.0.0-rc3      → ChatClientAgent / AgentSession
// ============================================================

using Azure;
using Azure.AI.OpenAI;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;

// ── 1. Load config from appsettings.json ─────────────────────
IConfiguration config = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

string endpoint        = config["AzureAI:Endpoint"]        ?? throw new InvalidOperationException("AzureAI:Endpoint missing");
string apiKey          = config["AzureAI:ApiKey"]          ?? throw new InvalidOperationException("AzureAI:ApiKey missing");
string modelDeployment = config["AzureAI:ModelDeployment"] ?? "o4-mini";

Console.WriteLine("==============================================");
Console.WriteLine("  MAF Learning – 01: Basic Agent Query");
Console.WriteLine("==============================================");
Console.WriteLine($"  Endpoint : {endpoint}");
Console.WriteLine($"  Model    : {modelDeployment}");
Console.WriteLine();

// ── 2. Create Azure OpenAI client (Microsoft.Extensions.AI) ──
//  AzureOpenAIClient connects to your Azure AI Foundry endpoint
//  .GetChatClient() returns an OpenAI.Chat.ChatClient for the deployment
//  .AsIChatClient() wraps it as an IChatClient abstraction
AzureOpenAIClient azureClient = new(
    new Uri(endpoint),
    new AzureKeyCredential(apiKey),
    new AzureOpenAIClientOptions(AzureOpenAIClientOptions.ServiceVersion.V2024_12_01_Preview)
);

IChatClient chatClient = azureClient.GetChatClient(modelDeployment).AsIChatClient();

// ── 3. Create a ChatClientAgent (Microsoft.Agents.AI) ────────
//  ChatClientAgent is the Microsoft.Agents.AI wrapper around
//  any IChatClient – adds session/memory/tool-call support
ChatClientAgent agent = new ChatClientAgent(
    chatClient: chatClient,
    name: "BasicLearningAgent",
    instructions: "You are a helpful AI assistant. Answer clearly and concisely."
);

// ── 4. Create a session ───────────────────────────────────────
//  ChatClientAgentSession tracks conversation history
//  (equivalent to a Thread in the Assistants API)
AgentSession session = await agent.CreateSessionAsync();

// ── 5. Send a query and get a response ───────────────────────
string userQuery = "What is the difference between an AI Agent and a regular LLM call?";

Console.WriteLine($"User  : {userQuery}");
Console.WriteLine();
Console.WriteLine("Agent : ");

AgentResponse response = await agent.RunAsync(
    message: userQuery,
    session: session
);

Console.WriteLine(response.Text);
Console.WriteLine();
Console.WriteLine("Done!");
