'use strict';
module.exports = function handler(req, res) {
  res.setHeader('Set-Cookie', [
    'sa_token=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0',
    'sa_user=; Path=/; Secure; SameSite=Lax; Max-Age=0'
  ]);
  res.redirect(302, '/');
};
