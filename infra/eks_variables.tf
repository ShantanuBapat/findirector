# infra/eks_variables.tf
#
# WHAT THIS DECLARES:
#   Input variables specific to the EKS (Kubernetes) layer — the tunable settings
#   referenced by the EKS module and Karpenter. Grouped in their own file so the
#   cluster knobs are easy to find, separate from the core and RDS variables.

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "findirector-eks"
}

variable "cluster_version" {
  description = "Kubernetes minor version for the EKS control plane"
  type        = string
  default     = "1.31"
}