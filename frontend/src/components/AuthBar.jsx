import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { cognitoMessage } from "../utils/cognitoMessage";
import "./AuthBar.css";

export default function AuthBar() {
  const {
    cognitoEnabled,
    loading,
    session,
    signIn,
    signUp,
    confirmSignUp,
    resendConfirmation,
    signOut,
  } = useAuth();

  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");
  const [code, setCode] = useState("");
  const [pendingConfirmEmail, setPendingConfirmEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  if (!cognitoEnabled) return null;

  if (loading) {
    return (
      <div className="authBar">
        <span className="authMuted">Account…</span>
      </div>
    );
  }

  if (session) {
    return (
      <div className="authBar authBarSignedIn">
        <span className="authEmail" title={session.email}>
          {session.email || "Signed in"}
        </span>
        <button type="button" className="authBtn authBtnGhost" onClick={() => signOut()}>
          Sign out
        </button>
      </div>
    );
  }

  const resetForm = () => {
    setError("");
    setPassword("");
    setPassword2("");
    setCode("");
  };

  const handleSignIn = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await signIn(email, password);
      setOpen(false);
      resetForm();
    } catch (err) {
      setError(cognitoMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();
    if (password !== password2) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await signUp(email, password);
      setPendingConfirmEmail(email.trim().toLowerCase());
      setPassword("");
      setPassword2("");
    } catch (err) {
      setError(cognitoMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const handleConfirm = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await confirmSignUp(pendingConfirmEmail || email, code);
      setPendingConfirmEmail("");
      setCode("");
      setMode("signin");
    } catch (err) {
      setError(cognitoMessage(err));
    } finally {
      setBusy(false);
    }
  };

  const handleResend = async () => {
    const target = pendingConfirmEmail || email.trim().toLowerCase();
    if (!target) return;
    setBusy(true);
    setError("");
    try {
      await resendConfirmation(target);
    } catch (err) {
      setError(cognitoMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="authBar">
      {!open ? (
        <button
          type="button"
          className="authBtn authBtnPrimary"
          onClick={() => {
            resetForm();
            setOpen(true);
            setMode("signin");
          }}
        >
          Sign in
        </button>
      ) : (
        <div className="authPanel">
          <div className="authPanelHeader">
            <span className="authPanelTitle">
              {pendingConfirmEmail ? "Confirm email" : mode === "signup" ? "Create account" : "Sign in"}
            </span>
            <button
              type="button"
              className="authClose"
              aria-label="Close"
              onClick={() => {
                setOpen(false);
                setPendingConfirmEmail("");
                resetForm();
              }}
            >
              ×
            </button>
          </div>

          {pendingConfirmEmail ? (
            <form className="authForm" onSubmit={handleConfirm}>
              <p className="authHint">
                Enter the verification code sent to <strong>{pendingConfirmEmail}</strong>.
              </p>
              <label className="authLabel">
                Code
                <input
                  className="authInput"
                  value={code}
                  onChange={(ev) => setCode(ev.target.value)}
                  autoComplete="one-time-code"
                  required
                />
              </label>
              {error ? <p className="authError">{error}</p> : null}
              <div className="authActions">
                <button type="submit" className="authBtn authBtnPrimary" disabled={busy}>
                  Confirm
                </button>
                <button type="button" className="authBtn authBtnGhost" onClick={handleResend} disabled={busy}>
                  Resend code
                </button>
              </div>
            </form>
          ) : mode === "signup" ? (
            <form className="authForm" onSubmit={handleSignUp}>
              <label className="authLabel">
                Email
                <input
                  className="authInput"
                  type="email"
                  value={email}
                  onChange={(ev) => setEmail(ev.target.value)}
                  autoComplete="email"
                  required
                />
              </label>
              <label className="authLabel">
                Password
                <input
                  className="authInput"
                  type="password"
                  value={password}
                  onChange={(ev) => setPassword(ev.target.value)}
                  autoComplete="new-password"
                  required
                  minLength={8}
                />
              </label>
              <label className="authLabel">
                Confirm password
                <input
                  className="authInput"
                  type="password"
                  value={password2}
                  onChange={(ev) => setPassword2(ev.target.value)}
                  autoComplete="new-password"
                  required
                  minLength={8}
                />
              </label>
              {error ? <p className="authError">{error}</p> : null}
              <div className="authActions">
                <button type="submit" className="authBtn authBtnPrimary" disabled={busy}>
                  Create account
                </button>
                <button
                  type="button"
                  className="authBtn authBtnGhost"
                  onClick={() => {
                    setMode("signin");
                    resetForm();
                  }}
                >
                  Back to sign in
                </button>
              </div>
            </form>
          ) : (
            <form className="authForm" onSubmit={handleSignIn}>
              <label className="authLabel">
                Email
                <input
                  className="authInput"
                  type="email"
                  value={email}
                  onChange={(ev) => setEmail(ev.target.value)}
                  autoComplete="username"
                  required
                />
              </label>
              <label className="authLabel">
                Password
                <input
                  className="authInput"
                  type="password"
                  value={password}
                  onChange={(ev) => setPassword(ev.target.value)}
                  autoComplete="current-password"
                  required
                />
              </label>
              {error ? <p className="authError">{error}</p> : null}
              <div className="authActions">
                <button type="submit" className="authBtn authBtnPrimary" disabled={busy}>
                  Sign in
                </button>
                <button
                  type="button"
                  className="authBtn authBtnGhost"
                  onClick={() => {
                    setMode("signup");
                    resetForm();
                  }}
                >
                  Create account
                </button>
              </div>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
