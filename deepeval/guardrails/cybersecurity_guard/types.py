from enum import Enum


class CyberattackType(Enum):
    BFLA = "BFLA"
    BOLA = "BOLA"
    DEBUG_ACCESS = "Debug Access"
    RBAC = "RBAC"
    SHELL_INJECTION = "Shell Injection"
    SQL_INJECTION = "SQL Injection"
    SSRF = "SSRF"


class GuardType(Enum):
    INPUT = "InputGuard"
    OUTPUT = "OutputGuard"
