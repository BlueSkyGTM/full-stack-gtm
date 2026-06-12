'use strict';
const { createJWT } = require('../_lib/auth');

module.exports = async function handler(req, res) {
  const { code } = req.query;
  if (!code) return res.status(400).send('Missing code');

  /* Exchange code for access token */
  const tokenRes = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify({
      client_id: process.env.GITHUB_CLIENT_ID,
      client_secret: process.env.GITHUB_CLIENT_SECRET,
      code
    })
  });
  const { access_token, error } = await tokenRes.json();
  if (error || !access_token) return res.status(401).send('OAuth failed');

  /* Get GitHub user */
  const userRes = await fetch('https://api.github.com/user', {
    headers: { Authorization: `Bearer ${access_token}`, 'User-Agent': 'synapse-academy' }
  });
  const { id, login, avatar_url } = await userRes.json();
  if (!id) return res.status(401).send('Could not fetch GitHub user');

  const jwt = createJWT({ sub: String(id), name: login, avatar: avatar_url });
  const maxAge = 30 * 24 * 3600;
  const secure = 'Secure; ';

  /* sa_token — httpOnly, carries the JWT */
  res.setHeader('Set-Cookie', [
    `sa_token=${jwt}; Path=/; HttpOnly; ${secure}SameSite=Lax; Max-Age=${maxAge}`,
    /* sa_user — readable by JS, just login:avatar (no secrets) */
    `sa_user=${encodeURIComponent(login + ':' + (avatar_url || ''))}; Path=/; ${secure}SameSite=Lax; Max-Age=${maxAge}`
  ]);
  res.redirect(302, '/');
};
