import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { cognitoMessage } from "../utils/cognitoMessage";
import "./AuthBar.css";
import "./AuthGateScreen.css";

export default function AuthGateScreen() {
  const { signIn, signUp, confirmSignUp, resendConfirmation } = useAuth();

  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [password2, setPassword2] = useState("");
  const [code, setCode] = useState("");
  const [pendingConfirmEmail, setPendingConfirmEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

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
      resetForm();
      setUsername("");
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
      await signUp(email, password, username);
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

  const title = pendingConfirmEmail
    ? "Confirm your email"
    : mode === "signup"
      ? "Create an account"
      : "Sign in";

  return (
    <div className="authGateRoot">
      <div className="authGateCard">
        <h1 className="authGateBrand">BlackjackAI</h1>
        <p className="authGateSubtitle">Sign in or register to use the card classifier.</p>
        <h2 className="authGateTitle">{title}</h2>

        {pendingConfirmEmail ? (
          <form className="authForm authGateForm" onSubmit={handleConfirm}>
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
            <div className="authActions authGateActions">
              <button type="submit" className="authBtn authBtnPrimary" disabled={busy}>
                Confirm
              </button>
              <button type="button" className="authBtn authBtnGhost" onClick={handleResend} disabled={busy}>
                Resend code
              </button>
            </div>
          </form>
        ) : mode === "signup" ? (
          <form className="authForm authGateForm" onSubmit={handleSignUp}>
            <label className="authLabel">
              Username
              <input
                className="authInput"
                type="text"
                value={username}
                onChange={(ev) => setUsername(ev.target.value)}
                autoComplete="username"
                required
              />
            </label>
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
            <div className="authActions authGateActions">
              <button type="submit" className="authBtn authBtnPrimary" disabled={busy}>
                Create account
              </button>
              <button
                type="button"
                className="authBtn authBtnGhost"
                onClick={() => {
                  setMode("signin");
                  resetForm();
                  setUsername("");
                }}
              >
                Back to sign in
              </button>
            </div>
          </form>
        ) : (
          <form className="authForm authGateForm" onSubmit={handleSignIn}>
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
            <div className="authActions authGateActions">
              <button type="submit" className="authBtn authBtnPrimary" disabled={busy}>
                Sign in
              </button>
              <button
                type="button"
                className="authBtn authBtnGhost"
                onClick={() => {
                  setMode("signup");
                  resetForm();
                  setUsername("");
                }}
              >
                Create account
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
