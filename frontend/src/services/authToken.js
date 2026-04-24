/** Set from AuthContext when session changes; read by axios in api.js. */
let idToken = null;

export function setAuthIdToken(token) {
  idToken = token || null;
}

export function getAuthIdToken() {
  return idToken;
}
