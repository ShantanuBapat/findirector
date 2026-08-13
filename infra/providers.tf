# infra/providers.tf
#
# WHAT THIS CONFIGURES:
#   The AWS provider — how Terraform talks to AWS — plus default_tags that get
#   auto-applied to EVERY resource this project creates.
#
# provider "aws" block:
#   region       : which AWS region to operate in (us-east-1, matching setup).
#   profile      : which local AWS CLI profile / credentials to use (the
#                  Identity Center SSO profile you log in with).
#   default_tags : a tag set stamped onto every taggable resource automatically,
#                  so you never hand-tag and never forget one. This is what makes
#                  "filter cost by Project" and "find everything to tear down"
#                  reliable.
#
# The tag VALUES reference variables (var.*) defined in variables.tf, so names
# like the project and environment aren't hardcoded here.

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }
  }
}

# --- Kubernetes + Helm providers: how Terraform talks to the CLUSTER ---
# These let Terraform install workloads (like the Karpenter controller) INTO the
# EKS cluster. They're configured from the EKS module's outputs — the cluster's
# control-plane API endpoint and a short-lived auth token — which resolve at
# apply time (after the cluster exists). Auth uses a token fetched via the AWS
# CLI, so no static kubeconfig is needed.

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--profile", var.aws_profile]
  }
}

provider "helm" {
  kubernetes = {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec = {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--profile", var.aws_profile]
    }
  }
}