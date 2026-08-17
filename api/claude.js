export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: { message: 'Method not allowed' } });
  }

  const key = process.env.GEMINI_API_KEY;
  if (!key) {
    return res.status(500).json({ error: { message: 'GEMINI_API_KEY not configured' } });
  }

  try {
    const { messages = [], system, max_tokens = 1000 } = req.body;

    // Anthropic role "assistant" → Gemini role "model"
    const contents = messages.map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: typeof m.content === 'string' ? m.content : m.content.map(c => c.text || '').join('') }]
    }));

    const geminiBody = {
      contents,
      generationConfig: { maxOutputTokens: max_tokens }
    };

    if (system) {
      geminiBody.systemInstruction = { parts: [{ text: system }] };
    }

    const upstream = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${key}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(geminiBody)
      }
    );

    const data = await upstream.json();

    if (!upstream.ok) {
      return res.status(upstream.status).json({
        error: { message: data.error?.message || `Gemini API ${upstream.status}` }
      });
    }

    // Return in Anthropic response shape so the frontend's existing error handling works unchanged
    const text = data.candidates?.[0]?.content?.parts?.map(p => p.text || '').join('') || '';
    return res.status(200).json({
      content: [{ type: 'text', text }]
    });

  } catch (err) {
    return res.status(502).json({
      error: { message: 'Upstream request failed: ' + err.message }
    });
  }
}
