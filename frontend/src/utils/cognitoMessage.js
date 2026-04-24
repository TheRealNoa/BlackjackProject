export function cognitoMessage(err) {
  if (!err) return "Something went wrong.";
  const code = err.code || err.name;
  const map = {
    UserNotConfirmedException: "Confirm your email with the code we sent you.",
    NotAuthorizedException: "Incorrect email or password.",
    UsernameExistsException: "An account with this email already exists.",
    InvalidPasswordException: err.message || "Password does not meet pool policy.",
    CodeMismatchException: "Invalid verification code.",
    ExpiredCodeException: "That code expired. Request a new one.",
    InvalidParameterException: err.message || "Invalid input.",
    LimitExceededException: "Too many attempts. Try again later.",
  };
  return map[code] || err.message || String(code);
}
