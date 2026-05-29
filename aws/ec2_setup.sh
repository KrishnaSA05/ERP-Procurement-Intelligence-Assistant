#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# aws/ec2_setup.sh
#
# Bootstraps a fresh EC2 t2.micro (Amazon Linux 2023 or Ubuntu 22.04)
# to run the full ERP Procurement Intelligence stack via Docker Compose.
#
# Run once after launching EC2:
#   chmod +x aws/ec2_setup.sh
#   ./aws/ec2_setup.sh
#
# Or pass as EC2 User Data (runs automatically on first boot):
#   Copy this script into the "User Data" field when launching EC2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
LOG="/var/log/ec2_setup.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo " ERP Procurement Intelligence — EC2 Setup"
echo " $(date)"
echo "=============================================="

# ── Detect OS ─────────────────────────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi
echo "OS detected: $OS"

# ── Install Docker ────────────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing Docker..."

if command -v docker &>/dev/null; then
    echo "  Docker already installed: $(docker --version)"
else
    if [[ "$OS" == "ubuntu" ]]; then
        apt-get update -q
        apt-get install -y ca-certificates curl gnupg
        install -m 0755 -d /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
            | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        chmod a+r /etc/apt/keyrings/docker.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
            https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
            | tee /etc/apt/sources.list.d/docker.list > /dev/null
        apt-get update -q
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    else
        # Amazon Linux 2023
        dnf update -y
        dnf install -y docker
        systemctl start docker
        systemctl enable docker
        # Install Docker Compose v2
        COMPOSE_VERSION="v2.24.5"
        mkdir -p /usr/local/lib/docker/cli-plugins
        curl -SL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
            -o /usr/local/lib/docker/cli-plugins/docker-compose
        chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    fi
    usermod -aG docker ec2-user 2>/dev/null || usermod -aG docker ubuntu 2>/dev/null || true
    echo "  ✓ Docker installed"
fi

systemctl start docker
systemctl enable docker

# ── Install Git ───────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Installing Git..."
if [[ "$OS" == "ubuntu" ]]; then
    apt-get install -y git
else
    dnf install -y git
fi
echo "  ✓ Git installed"

# ── Clone repository ──────────────────────────────────────────────────────────
echo ""
echo "[3/6] Cloning repository..."

APP_DIR="/opt/erp-procurement"
REPO_URL="${REPO_URL:-https://github.com/yourusername/erp-procurement-intelligence.git}"

if [ -d "$APP_DIR" ]; then
    echo "  Repo exists — pulling latest..."
    cd "$APP_DIR" && git pull origin main
else
    git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
echo "  ✓ Repository ready at $APP_DIR"

# ── Configure environment ─────────────────────────────────────────────────────
echo ""
echo "[4/6] Configuring environment..."

if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "  ⚠  .env created from .env.example"
    echo "  ⚠  IMPORTANT: Edit $APP_DIR/.env with your AWS credentials and RDS endpoint"
    echo "  ⚠  Then re-run: docker compose -f docker-compose.prod.yml up -d"
else
    echo "  ✓ .env already exists"
fi

# Set APP_ENV to production
sed -i 's/APP_ENV=development/APP_ENV=production/' "$APP_DIR/.env" || true

# ── Open firewall ports ───────────────────────────────────────────────────────
echo ""
echo "[5/6] Configuring firewall..."
# These are EC2 Security Group rules — configure in AWS console:
# Inbound rules needed:
#   Port 8000 (FastAPI)  — from your IP or 0.0.0.0/0 for demo
#   Port 8501 (Streamlit)— from your IP or 0.0.0.0/0 for demo
#   Port 22   (SSH)      — from your IP only
echo "  ⚠  Remember to open ports 8000 and 8501 in EC2 Security Group"

# ── Start services ────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Starting Docker services..."

cd "$APP_DIR"

# Build images
docker compose -f docker-compose.prod.yml build

# Start all services
docker compose -f docker-compose.prod.yml up -d

echo ""
echo "  Waiting 30s for services to initialise..."
sleep 30

# Health check
echo ""
echo "  Checking service health..."
docker compose -f docker-compose.prod.yml ps

# ── Done ──────────────────────────────────────────────────────────────────────
EC2_PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "YOUR_EC2_IP")

echo ""
echo "=============================================="
echo " SETUP COMPLETE"
echo "=============================================="
echo ""
echo " Application URLs:"
echo "   Streamlit UI  : http://${EC2_PUBLIC_IP}:8501"
echo "   FastAPI docs  : http://${EC2_PUBLIC_IP}:8000/docs"
echo "   Health check  : http://${EC2_PUBLIC_IP}:8000/health"
echo ""
echo " Next steps:"
echo "   1. Edit /opt/erp-procurement/.env with RDS endpoint + AWS creds"
echo "   2. Run: python aws/rds_setup.py --seed  (to populate RDS)"
echo "   3. Run: python aws/s3_upload.py --all   (to upload PDFs)"
echo "   4. Run: python src/ingestion/ingest_pipeline.py --all"
echo "   5. Restart: docker compose -f docker-compose.prod.yml restart api"
echo ""
echo " Logs:"
echo "   docker compose -f docker-compose.prod.yml logs -f api"
echo "   docker compose -f docker-compose.prod.yml logs -f streamlit"
echo "=============================================="
