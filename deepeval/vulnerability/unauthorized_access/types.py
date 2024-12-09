from enum import Enum


class UnauthorizedAccessType(Enum):
    BFLA = "Broken Function Level Authorization"
    BOLA = "Broken Object Level Authorization"
    RBAC = "Role-Based Access Control"
    DEBUG_ACCESS = "Debug Access"
    SHELL_INJECTION = "Shell Injection"
    SQL_INJECTION = "SQL Injection"
    SSRF = "Server Side Request Forgery"
