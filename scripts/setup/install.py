#!/usr/bin/env python3
"""
Inference Arena - Environment Setup Script

Cross-platform Python script to create .env configuration file.
Works on Windows, macOS, and Linux.

Usage:
    python scripts/setup_env.py

Author: Matthew Hong
"""

import secrets
import shutil
import sys
from getpass import getpass
from pathlib import Path
from typing import Optional


class Colors:
    """ANSI color codes (gracefully degrades on Windows CMD)."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color

    @classmethod
    def strip_if_windows(cls) -> None:
        """Remove colors if on Windows without ANSI support."""
        if sys.platform == "win32":
            # Windows 10+ supports ANSI, but older versions don't
            import os
            if not os.environ.get("ANSICON") and not os.environ.get("WT_SESSION"):
                cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.BOLD = cls.NC = ""


def print_header() -> None:
    """Print welcome header."""
    print(f"{Colors.BLUE}{'=' * 64}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}  Inference Arena - Environment Setup{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 64}{Colors.NC}\n")


def generate_secure_password(length: int = 32) -> str:
    """
    Generate a cryptographically secure password.

    Args:
        length: Password length (default: 32 characters)

    Returns:
        Secure random password string
    """
    # Use URL-safe characters (alphanumeric + -_)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def backup_existing_env(env_file: Path) -> Optional[Path]:
    """
    Create backup of existing .env file.

    Args:
        env_file: Path to .env file

    Returns:
        Path to backup file, or None if no backup needed
    """
    if not env_file.exists():
        return None

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = env_file.with_suffix(f".backup.{timestamp}")
    shutil.copy2(env_file, backup_file)
    return backup_file


def get_project_root() -> Path:
    """Get project root directory (parent of scripts/)."""
    return Path(__file__).parent.parent.resolve()


def check_gitignore(project_root: Path) -> None:
    """Ensure .env is in .gitignore."""
    gitignore_file = project_root / ".gitignore"

    if not gitignore_file.exists():
        print(f"{Colors.YELLOW}⚠ Warning: No .gitignore found{Colors.NC}")
        return

    gitignore_content = gitignore_file.read_text(encoding="utf-8")

    if ".env" not in gitignore_content or not any(
        line.strip() == ".env" for line in gitignore_content.splitlines()
    ):
        print(f"{Colors.YELLOW}⚠ Adding .env to .gitignore...{Colors.NC}")
        with gitignore_file.open("a", encoding="utf-8") as f:
            f.write("\n# Environment variables (secrets)\n")
            f.write(".env\n")
            f.write(".env.local\n")
            f.write(".env.*.local\n")
        print(f"{Colors.GREEN}✓ Updated .gitignore{Colors.NC}")


def setup_development_mode(env_example: Path, env_file: Path) -> None:
    """Setup with default development values."""
    print(f"{Colors.YELLOW}⚠ Using development defaults (NOT secure for production!){Colors.NC}")
    shutil.copy2(env_example, env_file)
    print(f"{Colors.GREEN}✓ Created .env with development defaults{Colors.NC}")


def setup_production_mode(env_example: Path, env_file: Path) -> None:
    """Setup with generated secure passwords."""
    print(f"{Colors.GREEN}Generating secure passwords...{Colors.NC}\n")

    # Copy template
    shutil.copy2(env_example, env_file)

    # Generate passwords
    minio_password = generate_secure_password()
    grafana_password = generate_secure_password()

    # Read current content
    content = env_file.read_text(encoding="utf-8")

    # Replace passwords
    content = content.replace(
        "MINIO_ROOT_PASSWORD=minioadmin",
        f"MINIO_ROOT_PASSWORD={minio_password}"
    )
    content = content.replace(
        "GRAFANA_ADMIN_PASSWORD=admin",
        f"GRAFANA_ADMIN_PASSWORD={grafana_password}"
    )

    # Write back
    env_file.write_text(content, encoding="utf-8")

    print(f"{Colors.GREEN}✓ Secure passwords generated!{Colors.NC}")
    print(f"{Colors.YELLOW}⚠ IMPORTANT: Save these passwords in a password manager!{Colors.NC}\n")
    print(f"{Colors.BOLD}MinIO Password:{Colors.NC}   {minio_password}")
    print(f"{Colors.BOLD}Grafana Password:{Colors.NC} {grafana_password}\n")


def setup_custom_mode(env_example: Path, env_file: Path) -> None:
    """Setup with user-provided values."""
    print(f"{Colors.BLUE}Custom configuration...{Colors.NC}\n")

    # Copy template
    shutil.copy2(env_example, env_file)

    # Get user input
    print("MinIO Configuration:")
    minio_user = input(f"  Username [{Colors.BLUE}minioadmin{Colors.NC}]: ").strip() or "minioadmin"
    minio_pass = getpass("  Password [leave blank for default]: ").strip()

    print("\nGrafana Configuration:")
    grafana_user = input(f"  Username [{Colors.BLUE}admin{Colors.NC}]: ").strip() or "admin"
    grafana_pass = getpass("  Password [leave blank for default]: ").strip()

    # Read current content
    content = env_file.read_text(encoding="utf-8")

    # Update values if provided
    if minio_user != "minioadmin":
        content = content.replace("MINIO_ROOT_USER=minioadmin", f"MINIO_ROOT_USER={minio_user}")

    if minio_pass:
        content = content.replace("MINIO_ROOT_PASSWORD=minioadmin", f"MINIO_ROOT_PASSWORD={minio_pass}")

    if grafana_user != "admin":
        content = content.replace("GRAFANA_ADMIN_USER=admin", f"GRAFANA_ADMIN_USER={grafana_user}")

    if grafana_pass:
        content = content.replace("GRAFANA_ADMIN_PASSWORD=admin", f"GRAFANA_ADMIN_PASSWORD={grafana_pass}")

    # Write back
    env_file.write_text(content, encoding="utf-8")

    print(f"\n{Colors.GREEN}✓ Custom configuration saved{Colors.NC}")


def print_next_steps() -> None:
    """Print next steps after setup."""
    print(f"\n{Colors.GREEN}{'=' * 64}{Colors.NC}")
    print(f"{Colors.GREEN}{Colors.BOLD}  ✓ Setup Complete!{Colors.NC}")
    print(f"{Colors.GREEN}{'=' * 64}{Colors.NC}\n")

    print("Next steps:")
    print("  1. Review your .env file")
    print("  2. Start infrastructure:")
    print(f"     {Colors.BLUE}docker compose -f infrastructure/docker-compose.infra.yml up -d{Colors.NC}")
    print("  3. Check services:")
    print(f"     {Colors.BLUE}docker ps{Colors.NC}\n")

    print("Service endpoints:")
    print("  MinIO Console:  http://localhost:9001")
    print("  Prometheus:     http://localhost:9090")
    print("  Grafana:        http://localhost:3000")
    print("  cAdvisor:       http://localhost:8080\n")

    print(f"{Colors.BLUE}For more information, see ENVIRONMENT.md{Colors.NC}\n")


def main() -> int:
    """Main setup workflow."""
    # Setup colors
    Colors.strip_if_windows()

    # Print header
    print_header()

    # Get paths
    project_root = get_project_root()
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"

    # Verify .env.example exists
    if not env_example.exists():
        print(f"{Colors.RED}✗ Error: .env.example not found at {env_example}{Colors.NC}")
        print("  Are you running this from the correct directory?")
        return 1

    # Handle existing .env
    if env_file.exists():
        print(f"{Colors.YELLOW}⚠ Warning: .env file already exists!{Colors.NC}\n")
        print("Options:")
        print("  1) Backup existing .env and create new one")
        print("  2) Keep existing .env (do nothing)")
        print("  3) Overwrite existing .env (dangerous!)\n")

        choice = input("Choose [1-3]: ").strip()

        if choice == "1":
            backup_file = backup_existing_env(env_file)
            if backup_file:
                print(f"{Colors.GREEN}✓ Backed up to: {backup_file}{Colors.NC}\n")
        elif choice == "2":
            print(f"{Colors.GREEN}✓ Keeping existing .env{Colors.NC}")
            return 0
        elif choice == "3":
            print(f"{Colors.YELLOW}⚠ Overwriting existing .env...{Colors.NC}\n")
        else:
            print(f"{Colors.RED}✗ Invalid choice. Exiting.{Colors.NC}")
            return 1

    # Choose configuration mode
    print(f"{Colors.BLUE}Configuration Mode:{Colors.NC}")
    print("  1) Development (use default passwords - NOT secure)")
    print("  2) Production (generate secure passwords)")
    print("  3) Custom (I'll enter passwords myself)\n")

    mode = input("Choose [1-3]: ").strip()
    print()

    if mode == "1":
        setup_development_mode(env_example, env_file)
    elif mode == "2":
        setup_production_mode(env_example, env_file)
    elif mode == "3":
        setup_custom_mode(env_example, env_file)
    else:
        print(f"{Colors.RED}✗ Invalid choice. Exiting.{Colors.NC}")
        return 1

    # Set permissions (Unix-like systems only)
    if hasattr(env_file, "chmod"):
        try:
            env_file.chmod(0o600)  # rw-------
            print(f"{Colors.GREEN}✓ Set .env permissions to 600 (owner read/write only){Colors.NC}")
        except (OSError, NotImplementedError):
            # Windows or permission error
            pass

    # Update .gitignore
    check_gitignore(project_root)

    # Print next steps
    print_next_steps()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}✗ Setup cancelled by user{Colors.NC}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}✗ Error: {e}{Colors.NC}")
        sys.exit(1)
