/**
 * Gemini AI Attribution Core
 * Synthesizes signals into causal narratives
 * Based on gemini_docs patterns
 */

import { GoogleGenAI } from '@google/genai';
import type {
    ThreatEvent,
    Signal,
    AttributionBrief,
    CausalFactor,
    SuggestedAction,
    ThreatId
} from '@/lib/types';

// === MODELS ===
const PRIMARY_MODEL = 'gemini-2.5-flash';
const FALLBACK_MODEL = 'gemini-2.5-flash-lite';

// === JSON SCHEMA FOR STRUCTURED OUTPUT ===
const AttributionBriefJsonSchema = {
    type: 'object',
    properties: {
        summary: {
            type: 'string',
            description: 'One-paragraph executive summary of what happened and why',
        },
        causes: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    factor: { type: 'string', description: 'The causal factor' },
                    impact: { type: 'string', enum: ['HIGH', 'MEDIUM', 'LOW'] },
                    evidence: { type: 'string', description: 'Evidence from signals' },
                },
                required: ['factor', 'impact', 'evidence'],
            },
        },
        suggestedActions: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    action: { type: 'string', description: 'Specific action to take' },
                    priority: { type: 'string', enum: ['IMMEDIATE', 'SOON', 'OPTIONAL'] },
                    expectedOutcome: { type: 'string', description: 'Expected outcome' },
                },
                required: ['action', 'priority', 'expectedOutcome'],
            },
        },
        confidence: {
            type: 'number',
            description: 'Confidence level 0-100',
        },
    },
    required: ['summary', 'causes', 'suggestedActions', 'confidence'],
};

// === GEMINI CLIENT ===
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
    if (!aiClient) {
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            throw new Error('GEMINI_API_KEY environment variable is not set');
        }
        aiClient = new GoogleGenAI({ apiKey });
    }
    return aiClient;
}

// === RETRY HELPER ===
async function withRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    delayMs: number = 1000
): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error as Error;
            const errorMessage = lastError.message || '';

            // Retry on transient errors (503, rate limit)
            if (errorMessage.includes('503') || errorMessage.includes('overloaded') || errorMessage.includes('rate')) {
                if (attempt < maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, delayMs * (attempt + 1)));
                    continue;
                }
            }
            throw error;
        }
    }

    throw lastError;
}

// === ATTRIBUTION CORE ===
// === ATTRIBUTION CORE ===
export async function generateAttributionBrief(
    threat: ThreatEvent,
    signals: Signal[]
): Promise<AttributionBrief> {
    // 1. CHECK CACHE
    try {
        const { getRedisClient } = await import('@/lib/redis/client'); // Lazy import
        const redis = getRedisClient();
        const cacheKey = `attribution:${threat.id}`;
        const cached = await redis.get(cacheKey);

        if (cached) {
            console.log('Attribution Cache Hit', threat.id);
            const parsedCache = JSON.parse(cached);
            return {
                ...parsedCache,
                generatedAt: new Date(parsedCache.generatedAt)
            };
        }
    } catch (e) {
        console.warn('Cache lookup failed, proceeding to API', e);
    }

    const ai = getGeminiClient();

    // Build context from signals
    const signalContext = signals.map(s =>
        `- ${s.type}: value=${s.value}, delta=${s.delta ?? 'N/A'}, time=${s.timestamp.toISOString()}`
    ).join('\n');

    const prompt = `
<role>
You are an Attribution Analyst for SellerOps, a marketplace seller intelligence system.
Your job is to explain WHY a threat occurred by synthesizing multiple signals.
</role>

<constraints>
1. Be precise and data-driven
2. Cite specific signals as evidence
3. Suggest actionable counter-moves
4. Verbosity: Medium - Be concise but thorough
5. DO NOT hallucinate causes not supported by signals
</constraints>

<context>
Threat Detected: ${threat.title}
Severity: ${threat.severity}
Description: ${threat.description}
Detection Time: ${threat.detectedAt.toISOString()}

Related Signals:
${signalContext}
</context>

<task>
Analyze the signals and provide an Attribution Brief explaining:
1. WHAT happened (summary)
2. WHY it happened (causal factors with evidence)
3. WHAT TO DO (suggested actions with priorities)
</task>
`;

    // Try primary model, fallback to lite on failure
    const callGemini = async (model: string) => {
        const response = await ai.models.generateContent({
            model,
            contents: prompt,
            config: {
                responseMimeType: 'application/json',
                responseSchema: AttributionBriefJsonSchema,
            },
        });

        const text = response.text;
        if (!text) {
            throw new Error('Empty response from Gemini');
        }

        return JSON.parse(text) as {
            summary: string;
            causes: CausalFactor[];
            suggestedActions: SuggestedAction[];
            confidence: number;
        };
    };

    let parsed;
    try {
        parsed = await withRetry(() => callGemini(PRIMARY_MODEL));
    } catch (error) {
        // Fallback to lite model
        console.warn('Primary model failed, trying fallback:', error);
        parsed = await withRetry(() => callGemini(FALLBACK_MODEL));
    }

    const result = {
        threatId: threat.id as ThreatId,
        summary: parsed.summary,
        causes: parsed.causes,
        suggestedActions: parsed.suggestedActions,
        confidence: parsed.confidence,
        generatedAt: new Date(),
    };

    // 2. SAVE TO CACHE (24 Hours)
    try {
        const { getRedisClient } = await import('@/lib/redis/client');
        const redis = getRedisClient();
        const cacheKey = `attribution:${threat.id}`;
        await redis.setex(cacheKey, 86400, JSON.stringify(result)); // 24h
    } catch (e) {
        console.error('Failed to cache attribution:', e);
    }

    return result;
}

// === STREAMING ATTRIBUTION (For real-time typing effect) ===
export async function* streamAttributionBrief(
    threat: ThreatEvent,
    signals: Signal[]
): AsyncGenerator<string, void, unknown> {
    const ai = getGeminiClient();

    const signalContext = signals.map(s =>
        `- ${s.type}: value=${s.value}, delta=${s.delta ?? 'N/A'}`
    ).join('\n');

    const prompt = `
You are an Attribution Analyst. Write a brief tactical report explaining this threat:

Threat: ${threat.title} (${threat.severity})
Signals:
${signalContext}

Format: Start with "THREAT ANALYSIS:" then explain causes and suggest actions.
Keep it under 200 words. Be direct and tactical.
`;

    const stream = await ai.models.generateContentStream({
        model: FALLBACK_MODEL, // Use lite for streaming (faster)
        contents: prompt,
    });

    for await (const chunk of stream) {
        const text = chunk.candidates?.[0]?.content?.parts?.[0]?.text;
        if (text) {
            yield text;
        }
    }
}
