# infra/rds.tf
#
# WHAT THIS CREATES:
#   The RDS (managed Postgres + pgvector) database layer, in the private subnets
#   of the VPC. Built in dependency order: subnet group (where it lives) ->
#   security group (who may connect) -> credentials (Secrets Manager) -> the DB
#   instance itself -> outputs.
#
# This file is added to incrementally; each resource is annotated with its role.

# --- DB subnet group: which subnets RDS may place the database in ---
# RDS requires subnets in >= 2 AZs. We hand it the VPC's two PRIVATE subnets
# (referenced from the VPC module's output, so we never hardcode subnet IDs),
# keeping the database off the internet.
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name = "${var.project_name}-db-subnet-group"
  }
}

# --- Client security group: the "badge" for things allowed to reach the DB ---
# Anything wearing this SG (the SSM instance now; API pods later) may connect to
# the database. We reference this SG from the DB's inbound rule, rather than IPs,
# because SG membership is stable while IPs churn.
resource "aws_security_group" "db_client" {
  name        = "${var.project_name}-db-client"
  description = "Attached to resources allowed to connect to the RDS database"
  vpc_id      = module.vpc.vpc_id

  # Allow all outbound (so the client can actually reach the DB and the internet
  # for package installs etc.). Egress is commonly left open; ingress is the
  # tightly-controlled direction.
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-db-client"
  }
}

# --- DB security group: the firewall ON the database ---
# Allows inbound Postgres (5432) ONLY from resources wearing the db_client SG.
# Nothing else in the VPC — let alone the internet — can reach the database.
resource "aws_security_group" "db" {
  name        = "${var.project_name}-db"
  description = "Firewall for the RDS database; allows Postgres only from db_client"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "Postgres from db_client SG only"
    from_port       = var.db_port
    to_port         = var.db_port
    protocol        = "tcp"
    security_groups = [aws_security_group.db_client.id]
  }

  tags = {
    Name = "${var.project_name}-db"
  }
}

# --- Credentials: generate a random master password, store in Secrets Manager ---
# 1) random_password generates a strong password at plan time (it lives only in
#    Terraform state, which is in the encrypted S3 backend — never in .tf files).
resource "random_password" "db_master" {
  length  = 24
  special = true
  # Exclude characters RDS disallows in passwords (/, @, ", and space).
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

# 2) The Secrets Manager secret — a named container for the credential.
resource "aws_secretsmanager_secret" "db_master" {
  name        = "${var.project_name}/db/master-credentials"
  description = "Master credentials for the FinDirector RDS Postgres instance"

  tags = {
    Name = "${var.project_name}-db-master-credentials"
  }
}

# 3) The secret VERSION — the actual value stored inside the container, as JSON
#    (username + password), so anything connecting can fetch both from one secret.
resource "aws_secretsmanager_secret_version" "db_master" {
  secret_id = aws_secretsmanager_secret.db_master.id
  secret_string = jsonencode({
    username = var.db_master_username
    password = random_password.db_master.result
  })
}

# --- The RDS instance: managed Postgres + pgvector, in the private subnets ---
# Wires together everything above: lives in the db_subnet_group (private subnets),
# firewalled by the db security group, credentialed from the generated password.
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-db"

  # Engine
  engine         = "postgres"
  engine_version = var.db_engine_version

  # Size and storage
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  storage_type      = "gp3"
  storage_encrypted = true

  # The initial database + master credentials (from the generated password).
  db_name  = var.db_name
  username = var.db_master_username
  password = random_password.db_master.result
  port     = var.db_port

  # Placement + access: private subnets, firewalled to the db security group.
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  publicly_accessible    = false
  multi_az               = false

  # Backups / teardown (belt-and-suspenders): keep automated backups, and take a
  # final snapshot on destroy so a teardown doesn't lose the corpus.
  backup_retention_period   = 1
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.project_name}-db-final-${formatdate("YYYYMMDDhhmmss", timestamp())}"

  # Allow minor-version auto-upgrades; deletion protection off (dev, we destroy).
  auto_minor_version_upgrade = true
  deletion_protection        = false

  tags = {
    Name = "${var.project_name}-db"
  }
}