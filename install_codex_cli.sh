#!/usr/bin/env bash
# ============================================================
# Codex CLI 一键安装脚本（网络自适应：公网直连 or 科研网代理）
# 适用: Ubuntu x86_64 (root)
# 用法: bash install_codex_cli.sh
# 流程: 网络预检(直连?) -> 不通则交互输入科研网代理 -> 装 Node -> npm 装 @openai/codex -> 验证
# ============================================================
set -euo pipefail

# ---------- [1/5] 网络预检: 公网直连检测 ----------
echo "==> [1/5] 公网直连检测 (registry.npmjs.org)"
PROXY_MODE=0
if env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
   timeout 10 curl -sI --noproxy "*" -o /dev/null -w "%{http_code}" https://registry.npmjs.org/ 2>/dev/null | grep -q 200; then
  echo "    ✅ 公网直连可用 → 采用无代理安装"
  # 强制清掉环境里可能残留的代理，确保全程直连
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
else
  echo "    ❌ 无法直连公网（无外网出口或 DNS 受限）"
  echo "==> 该机需要科研网代理才能访问公网，请输入代理信息："
  read -r -p "    代理 IP 地址 (如 114.111.17.35): " PROXY_IP
  read -r -p "    代理端口 [默认 3128]: " PROXY_PORT; PROXY_PORT=${PROXY_PORT:-3128}
  read -r -p "    代理账号: " PROXY_USER
  read -r -s -p "    代理密码: " PROXY_PASS; echo ""
  [ -n "${PROXY_IP:-}" ]   || { echo "❌ 代理 IP 必填"; exit 1; }
  [ -n "${PROXY_USER:-}" ] || { echo "❌ 代理账号必填"; exit 1; }
  # URL 编码账号密码（防 @ : / 等特殊字符破坏代理串）
  ENC_USER=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$PROXY_USER" 2>/dev/null || printf '%s' "$PROXY_USER")
  ENC_PASS=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "$PROXY_PASS" 2>/dev/null || printf '%s' "$PROXY_PASS")
  PROXY_URL="http://${ENC_USER}:${ENC_PASS}@${PROXY_IP}:${PROXY_PORT}"
  export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL"
  export all_proxy="$PROXY_URL" ALL_PROXY="$PROXY_URL"
  export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
  # npm 也走该代理
  export npm_config_proxy="$PROXY_URL" npm_config_https_proxy="$PROXY_URL"
  echo "    代理已配置，正在验证连通性…"
  if ! timeout 15 curl -sI -o /dev/null -w "%{http_code}" https://registry.npmjs.org/ 2>/dev/null | grep -q 200; then
    echo "❌ 科研网代理连接失败（IP/端口/账号/密码有误或不可达），已退出"
    exit 1
  fi
  echo "    ✅ 科研网代理连通 → 采用代理安装"
  PROXY_MODE=1
fi

# ---------- [2/5] 确保 Node.js v22 LTS ----------
echo "==> [2/5] 检查 Node.js"
NEED_NODE=0
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR=$(node -v | sed 's/^v//; s/\..*//')
  echo "    已检测到 Node $(node -v)（主版本 $NODE_MAJOR）"
  if [ "$NODE_MAJOR" -lt 18 ]; then
    echo "    ⚠️ 版本过低（Codex CLI 需 Node >= 18），将升级到 v22 LTS"
    NEED_NODE=1
  fi
else
  echo "    未检测到 Node，开始安装 v22 LTS"
  NEED_NODE=1
fi

if [ "$NEED_NODE" -eq 1 ]; then
  NODE_URL_BASE="https://nodejs.org/dist/latest-v22.x"
  EXT=xz
  command -v xz >/dev/null 2>&1 || EXT=gz
  FN=$(curl -fsSL --retry 3 --connect-timeout 15 "$NODE_URL_BASE/" | grep -oE "node-v[0-9]+\.[0-9]+\.[0-9]+-linux-x64\.tar\.$EXT" | head -1)
  [ -n "$FN" ] || { echo "❌ 无法从 nodejs.org 解析最新版本号（网络异常?）"; exit 1; }
  echo "    下载 $FN（$([ "$PROXY_MODE" -eq 1 ] && echo 走科研网代理 || echo 直连 nodejs.org)）"
  curl -fL --retry 3 --connect-timeout 15 -o "/tmp/$FN" "$NODE_URL_BASE/$FN"
  echo "    解压安装到 /usr/local"
  if [ "$EXT" = "xz" ]; then
    tar -xJf "/tmp/$FN" -C /usr/local --strip-components=1
  else
    tar -xzf "/tmp/$FN" -C /usr/local --strip-components=1
  fi
  rm -f "/tmp/$FN"
  hash -r
  echo "    Node: $(node -v) | npm: $(npm -v)"
fi

# ---------- [3/5] npm 全局安装 ----------
echo "==> [3/5] npm 全局安装 @openai/codex（$([ "$PROXY_MODE" -eq 1 ] && echo 走科研网代理 || echo 直连 registry.npmjs.org)）"
npm install -g @openai/codex@latest --no-fund --no-audit

# ---------- [4/5] 验证 ----------
echo "==> [4/5] 验证安装"
command -v codex
codex --version

# ---------- [5/5] 完成 ----------
echo ""
echo "✅ Codex CLI 安装完成"
echo "   提示: 首次使用需运行 codex login 认证"
if [ "$PROXY_MODE" -eq 1 ]; then
  echo "   ⚠️ 本机通过科研网代理安装。若日常使用 codex 需联网，请保留代理环境变量或自行配置。"
fi
