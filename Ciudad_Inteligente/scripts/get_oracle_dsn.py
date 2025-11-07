#!/usr/bin/env python3
"""
Descubre y muestra el DSN de Oracle (host:port/service_name) a partir de un tnsnames.ora.

Uso:
  python scripts/get_oracle_dsn.py               # Muestra el primer DSN encontrado o ORACLE_DSN si existe
  python scripts/get_oracle_dsn.py --list        # Lista aliases + DSN
  python scripts/get_oracle_dsn.py --alias NAME  # Muestra DSN para el alias NAME
  python scripts/get_oracle_dsn.py --tns PATH    # Especifica ruta a tnsnames.ora (por defecto: ./wallet/tnsnames.ora o %TNS_ADMIN%/tnsnames.ora)
  python scripts/get_oracle_dsn.py --json        # Imprime JSON con {alias: dsn}

Nota: Si existe la variable de entorno ORACLE_DSN, se imprime y sale con código 0.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import Dict, Optional, Tuple

AliasMap = Dict[str, Tuple[str, Optional[str]]]
# alias -> (dsn, protocol)


def find_tnsnames(default_root: str) -> Optional[str]:
    # 1) -- TNS_ADMIN env
    tns_admin = os.environ.get("TNS_ADMIN")
    if tns_admin:
        p = os.path.join(tns_admin, "tnsnames.ora")
        if os.path.isfile(p):
            return p
    # 2) -- ./wallet/tnsnames.ora
    wallet = os.path.join(default_root, "wallet", "tnsnames.ora")
    if os.path.isfile(wallet):
        return wallet
    # 3) -- ./tnsnames.ora
    local = os.path.join(default_root, "tnsnames.ora")
    if os.path.isfile(local):
        return local
    return None


def parse_tnsnames(path: str) -> AliasMap:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # remove comments starting with # or ;
    clean = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";"):
            continue
        clean.append(ln.rstrip("\n"))

    aliases: AliasMap = {}
    i = 0
    while i < len(clean):
        line = clean[i]
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*(.*)$", line)
        if not m:
            i += 1
            continue
        alias = m.group(1)
        rhs = m.group(2).strip()
        # acumular hasta balancear paréntesis
        body = rhs
        depth = rhs.count("(") - rhs.count(")")
        i += 1
        while depth > 0 and i < len(clean):
            body += " " + clean[i].strip()
            depth += clean[i].count("(") - clean[i].count(")")
            i += 1
        # extraer host, port, service_name, protocol
        host = _search(body, r"\bhost\s*=\s*([^)\s]+)")
        port = _search(body, r"\bport\s*=\s*([0-9]+)")
        svc = _search(body, r"\bservice_name\s*=\s*([^)\s]+)")
        if not svc:
            # fallback SID
            svc = _search(body, r"\bSID\s*=\s*([^)\s]+)")
        proto = _search(body, r"\bprotocol\s*=\s*([^)\s]+)")
        if host and port and svc:
            dsn = f"{host}:{port}/{svc}"
            aliases[alias] = (dsn, proto)
        else:
            # alias sin info completa, lo omitimos
            pass
    return aliases


def _search(text: str, pattern: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.I)
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Descubre DSN desde tnsnames.ora")
    parser.add_argument("--tns", dest="tns_path", help="Ruta a tnsnames.ora")
    parser.add_argument("--alias", dest="alias", help="Alias a resolver (ej: hackathon_high)")
    parser.add_argument("--list", dest="list_mode", action="store_true", help="Listar todos los aliases")
    parser.add_argument("--json", dest="json_mode", action="store_true", help="Salida en JSON")
    args = parser.parse_args()

    # Si ya hay ORACLE_DSN en el entorno, mostrar y salir
    env_dsn = os.environ.get("ORACLE_DSN")
    if env_dsn and not args.list_mode and not args.json_mode and not args.alias:
        print(env_dsn)
        return 0

    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, os.pardir))  # repo root

    path = args.tns_path or find_tnsnames(root)
    if not path:
        print("ERROR: No se encontró tnsnames.ora. Usa --tns PATH o define TNS_ADMIN o coloca wallet/tnsnames.ora", file=sys.stderr)
        return 2

    aliases = parse_tnsnames(path)
    if not aliases:
        print(f"ERROR: No se encontraron aliases válidos en {path}", file=sys.stderr)
        return 3

    if args.list_mode or args.json_mode:
        if args.json_mode:
            data = {k: v[0] for k, v in aliases.items()}
            print(json.dumps(data, indent=2))
        else:
            for name, (dsn, proto) in aliases.items():
                suffix = f" [{proto}]" if proto else ""
                print(f"{name}: {dsn}{suffix}")
        return 0

    if args.alias:
        if args.alias in aliases:
            print(aliases[args.alias][0])
            return 0
        print(f"ERROR: Alias '{args.alias}' no encontrado en {path}", file=sys.stderr)
        return 4

    # Por defecto, devolver el primero
    first_alias = next(iter(aliases))
    print(aliases[first_alias][0])
    return 0


if __name__ == "__main__":
    sys.exit(main())

