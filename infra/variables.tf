# infra/variables.tf
#
# WHAT THIS DECLARES:
#   Input variables for the project — the tunable settings referenced elsewhere
#   as var.<name>. Each has a type, a description, and a default. Centralizing
#   them means all the project's knobs live in one reviewable place instead of
#   being hardcoded across resource files.
#
# Each variable block:
#   description : human explanation (shows in `terraform` prompts and docs).
#   type        : the expected type (string here) — Terraform validates it.
#   default     : the value used when none is supplied explicitly.

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Local AWS CLI profile (Identity Center SSO) Terraform authenticates with"
  type        = string
  default     = "findirector"
}

variable "project_name" {
  description = "Project name — used in resource names and the Project tag"
  type        = string
  default     = "findirector"
}

variable "environment" {
  description = "Deployment environment (dev / staging / prod) — used in the Environment tag"
  type        = string
  default     = "dev"
}

variable "owner" {
  description = "Resource owner — used in the Owner tag"
  type        = string
  default     = "shantanu"
}