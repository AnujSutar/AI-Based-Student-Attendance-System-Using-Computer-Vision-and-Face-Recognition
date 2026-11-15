"""Simple local admin authentication for the college prototype."""

from datetime import datetime

from database import get_connection, hash_password, verify_password


def authenticate(username: str, password: str) -> bool:
    """Verify an admin username and password against its salted stored hash."""
    with get_connection() as connection:
        admin = connection.execute(
            "SELECT password_hash FROM admin WHERE username = ?", (username.strip(),)
        ).fetchone()
    return bool(admin and verify_password(password, admin["password_hash"]))


def change_password(username: str, current_password: str, new_password: str) -> tuple[bool, str]:
    """Change the logged-in admin password after current-password verification."""
    if len(new_password) < 6:
        return False, "Use a password with at least 6 characters."
    if not authenticate(username, current_password):
        return False, "Current password is incorrect."
    with get_connection() as connection:
        connection.execute(
            "UPDATE admin SET password_hash = ?, created_at = ? WHERE username = ?",
            (hash_password(new_password), datetime.now().isoformat(timespec="seconds"), username),
        )
    return True, "Password changed successfully."
