'use strict';
const { getUser, redisGet, redisSet } = require('./_lib/auth');

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', process.env.SITE_URL || '*');
  res.setHeader('Access-Control-Allow-Credentials', 'true');

  const user = getUser(req);
  if (!user) return res.status(401).json({ error: 'Not authenticated' });

  const key = `sa:progress:${user.sub}`;

  if (req.method === 'GET') {
    const data = await redisGet(key);
    console.log('[progress] GET', key, data ? 'found' : 'empty');
    return res.json(data || { v: 1, done: {}, days: [], updatedAt: 0 });
  }

  if (req.method === 'POST') {
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    if (!body || typeof body !== 'object') {
      console.error('[progress] POST bad body:', JSON.stringify(body));
      return res.status(400).json({ error: 'Invalid body' });
    }
    console.log('[progress] POST', key, 'done keys:', Object.keys(body.done || {}).length);
    await redisSet(key, body);
    return res.json({ ok: true });
  }

  res.status(405).json({ error: 'Method not allowed' });
};
