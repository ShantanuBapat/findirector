# infra/versions.tf
#
# WHAT THIS DECLARES:
#   The Terraform version and provider versions this project requires. Pinning
#   these makes builds reproducible — future-you, a teammate, or CI all use the
#   same tooling and get the same behavior (like requirements.txt for infra).
#
# terraform.required_version : the minimum Terraform CLI version.
# required_providers         : which providers to download and their version
#                              constraints. `hashicorp/aws` is the plugin that
#                              talks to AWS. `~> 5.0` means ">= 5.0 and < 6.0"
#                              (allow patch/minor updates, but not a major bump
#                              that could break things).

terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}