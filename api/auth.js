'use strict';
/* Kick off GitHub OAuth */
module.exports = function handler(req, res) {
  const params = new URLSearchParams({
    client_id: process.env.GITHUB_CLIENT_ID,
    redirect_uri: `${process.env.SITE_URL}/api/auth/callback`,
    scope: 'read:user',
    state: require('crypto').randomBytes(8).toString('hex')
  });
  res.redirect(302, `https://github.com/login/oauth/authorize?${params}`);
};
