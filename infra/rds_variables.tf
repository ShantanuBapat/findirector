# infra/rds_variables.tf
#
# WHAT THIS DECLARES:
#   Input variables specific to the RDS (database) layer — the tunable settings
#   referenced by the RDS resources. Grouped in their own file so the database
#   knobs are easy to find and review, separate from the core project variables.
#
# Each has a type, description, and a sensible default for a small dev database.

variable "db_name" {
  description = "The initial database name created inside the Postgres instance"
  type        = string
  default     = "findirector"
}

variable "db_master_username" {
  description = "Master (admin) username for the Postgres instance"
  type        = string
  default     = "findirector_admin"
}

variable "db_engine_version" {
  description = "Postgres engine major.minor version for RDS"
  type        = string
  default     = "16.4"
}

variable "db_instance_class" {
  description = "RDS instance size (compute/memory). db.t3.micro is smallest/cheapest for dev"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "Allocated storage in GB (your corpus is small; 20 GB is the RDS minimum for gp3)"
  type        = number
  default     = 20
}

variable "db_port" {
  description = "Port Postgres listens on"
  type        = number
  default     = 5432
}