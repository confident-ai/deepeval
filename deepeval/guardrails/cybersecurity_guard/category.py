from enum import Enum


class CyberattackCategory(Enum):
    BFLA = "BFLA"
    BOLA = "BOLA"
    DEBUG_ACCESS = "Debug Access"
    RBAC = "RBAC"
    SHELL_INJECTION = "Shell Injection"
    SQL_INJECTION = "SQL Injection"
    SSRF = "SSRF"
