# infra/outputs.tf
#
# WHAT THIS DECLARES:
#   Output values surfaced after `terraform apply` — the pieces you need to
#   connect to and use the infrastructure. Outputs expose generated values (like
#   the RDS endpoint, known only after creation) without digging through state.

# RDS connection endpoint (host:port) — what a client points at (via the SSM
# tunnel). Known only after the instance is created.
output "db_endpoint" {
  description = "RDS instance endpoint (host:port)"
  value       = aws_db_instance.main.endpoint
}

# The hostname alone (no port) — the SSM port-forward target.
output "db_address" {
  description = "RDS instance hostname"
  value       = aws_db_instance.main.address
}

# The initial database name.
output "db_name" {
  description = "Initial database name"
  value       = aws_db_instance.main.db_name
}

# The Secrets Manager secret NAME holding the master credentials. Fetch the
# username/password from here when connecting — never from code.
output "db_secret_name" {
  description = "Secrets Manager secret holding the DB master credentials"
  value       = aws_secretsmanager_secret.db_master.name
}

# VPC id — handy reference for the SSM instance we add next.
output "vpc_id" {
  description = "VPC id"
  value       = module.vpc.vpc_id
}

# --- EKS outputs ---

# The command to configure kubectl to talk to this cluster. Run this after apply
# to point your local kubectl at the cluster's control-plane API server.
output "configure_kubectl" {
  description = "Run this to point kubectl at the cluster"
  value       = "aws eks update-kubeconfig --name ${module.eks.cluster_name} --region ${var.aws_region} --profile ${var.aws_profile}"
}

# The cluster's control-plane API server endpoint.
output "cluster_endpoint" {
  description = "EKS control-plane API server endpoint"
  value       = module.eks.cluster_endpoint
}

# The cluster name.
output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}