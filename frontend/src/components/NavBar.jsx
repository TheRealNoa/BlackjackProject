import AuthBar from "./AuthBar";
import "./NavBar.css";

function NavBar() {
  return (
    <header className="nav">
      <div className="nav-left">
        <span className="nav-brand">BlackjackAI</span>
      </div>
      <div className="nav-right">
        <span className="nav-pill">Card Classifier</span>
        <AuthBar />
      </div>
    </header>
  );
}

export default NavBar;
