import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  AuthenticationDetails,
  CognitoUser,
  CognitoUserPool,
} from "amazon-cognito-identity-js";
import { setAuthIdToken } from "../services/authToken";

const region = (import.meta.env.VITE_COGNITO_REGION ?? "").trim();
const userPoolId = (import.meta.env.VITE_COGNITO_USER_POOL_ID ?? "").trim();
const clientId = (import.meta.env.VITE_COGNITO_CLIENT_ID ?? "").trim();

export const cognitoEnabled = Boolean(userPoolId && clientId);

function getPool() {
  if (!cognitoEnabled) return null;
  return new CognitoUserPool({ UserPoolId: userPoolId, ClientId: clientId });
}

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(cognitoEnabled);

  const applySessionFromCognito = useCallback((s) => {
    if (!s?.isValid()) {
      setSession(null);
      setAuthIdToken(null);
      return;
    }
    const idTok = s.getIdToken().getJwtToken();
    const accessTok = s.getAccessToken().getJwtToken();
    const refreshTok = s.getRefreshToken()?.getToken() ?? "";
    let email = "";
    try {
      const payload = s.getIdToken().decodePayload();
      email = payload.email || payload["cognito:username"] || "";
    } catch {
      email = "";
    }
    const next = { email, idToken: idTok, accessToken: accessTok, refreshToken: refreshTok };
    setSession(next);
    setAuthIdToken(idTok);
  }, []);

  const refreshSession = useCallback(() => {
    const p = getPool();
    if (!p) {
      setLoading(false);
      setAuthIdToken(null);
      return;
    }
    const cu = p.getCurrentUser();
    if (!cu) {
      setSession(null);
      setAuthIdToken(null);
      setLoading(false);
      return;
    }
    cu.getSession((err, s) => {
      setLoading(false);
      if (err || !s) {
        setSession(null);
        setAuthIdToken(null);
        return;
      }
      applySessionFromCognito(s);
    });
  }, [applySessionFromCognito]);

  useEffect(() => {
    refreshSession();
  }, [refreshSession]);

  const signIn = useCallback(
    (email, password) =>
      new Promise((resolve, reject) => {
        const p = getPool();
        if (!p) {
          reject(new Error("Cognito is not configured"));
          return;
        }
        const username = email.trim().toLowerCase();
        const authDetails = new AuthenticationDetails({
          Username: username,
          Password: password,
        });
        const user = new CognitoUser({ Username: username, Pool: p });
        user.authenticateUser(authDetails, {
          onSuccess: (s) => {
            applySessionFromCognito(s);
            resolve();
          },
          onFailure: (err) => reject(err),
          newPasswordRequired: () =>
            reject(new Error("Password change required.")),
        });
      }),
    [applySessionFromCognito]
  );

  const signUp = useCallback(
    (email, password) =>
      new Promise((resolve, reject) => {
        const p = getPool();
        if (!p) {
          reject(new Error("Cognito is not configured"));
          return;
        }
        const username = email.trim().toLowerCase();
        p.signUp(
          username,
          password,
          [{ Name: "email", Value: username }],
          null,
          (err, result) => {
            if (err) reject(err);
            else resolve(result);
          }
        );
      }),
    []
  );

  const confirmSignUp = useCallback(
    (email, code) =>
      new Promise((resolve, reject) => {
        const p = getPool();
        if (!p) {
          reject(new Error("Cognito is not configured"));
          return;
        }
        const user = new CognitoUser({
          Username: email.trim().toLowerCase(),
          Pool: p,
        });
        user.confirmRegistration(code.trim(), true, (err, ok) => {
          if (err) reject(err);
          else resolve(ok);
        });
      }),
    []
  );

  const resendConfirmation = useCallback(
    (email) =>
      new Promise((resolve, reject) => {
        const p = getPool();
        if (!p) {
          reject(new Error("Cognito is not configured"));
          return;
        }
        const user = new CognitoUser({
          Username: email.trim().toLowerCase(),
          Pool: p,
        });
        user.resendConfirmationCode((err, ok) => {
          if (err) reject(err);
          else resolve(ok);
        });
      }),
    []
  );

  const signOut = useCallback(() => {
    const p = getPool();
    const cu = p?.getCurrentUser();
    if (cu) cu.signOut();
    setSession(null);
    setAuthIdToken(null);
  }, []);

  const value = useMemo(
    () => ({
      cognitoEnabled,
      cognitoRegion: region,
      loading,
      session,
      signIn,
      signUp,
      confirmSignUp,
      resendConfirmation,
      signOut,
      refreshSession,
    }),
    [
      loading,
      session,
      signIn,
      signUp,
      confirmSignUp,
      resendConfirmation,
      signOut,
      refreshSession,
    ]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
