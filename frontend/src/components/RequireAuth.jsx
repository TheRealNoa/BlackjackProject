import { useAuth } from "../context/AuthContext";
import AuthGateScreen from "./AuthGateScreen";
import "./RequireAuth.css";

/**
 * When Cognito env vars are set, blocks the app until the user has a valid session.
 * When Cognito is not configured, children render unchanged (local dev without auth).
 */
export default function RequireAuth({ children }) {
  const { cognitoEnabled, loading, session } = useAuth();

  if (!cognitoEnabled) {
    return children;
  }

  if (loading) {
    return (
      <div className="authGateRoot">
        <div className="authGateLoading">
          <p className="authGateLoadingText">Checking sign-in…</p>
        </div>
      </div>
    );
  }

  if (!session) {
    return <AuthGateScreen />;
  }

  return children;
}
