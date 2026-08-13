# infra/eks.tf
#
# WHAT THIS CREATES:
#   The EKS cluster (managed Kubernetes control plane) via the official EKS
#   module, plugged into the existing VPC's private subnets. Includes a small
#   managed node group to bootstrap cluster add-ons (notably Karpenter), which
#   then provisions all other nodes on demand.

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version

  # Give the cluster a public endpoint (so you can reach it with kubectl from
  # your Mac) AND a private one (so in-VPC nodes reach it privately).
  cluster_endpoint_public_access = true

  # Plug into the existing VPC: nodes live in the private subnets.
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Grant the Terraform-running identity (your SSO admin) cluster-admin access,
  # so you can use kubectl right after creation.
  enable_cluster_creator_admin_permissions = true

  # A minimal managed node group to bootstrap add-ons (Karpenter runs here).
  # Karpenter provisions everything else on demand.
  eks_managed_node_groups = {
    bootstrap = {
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 3
      desired_size   = 2
    }
  }

  tags = {
    Name = var.cluster_name
    # Karpenter discovers subnets/security groups by this tag (used later).
    "karpenter.sh/discovery" = var.cluster_name
  }
}

# --- Karpenter IAM + queue (via the official Karpenter sub-module) ---
# Builds the permissions Karpenter needs to launch/terminate EC2 on your behalf,
# the IAM role that Karpenter-provisioned nodes will wear, and an interruption
# queue (so Karpenter reacts to EC2 spot-reclaim / maintenance events).
module "karpenter" {
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 20.0"

  cluster_name = module.eks.cluster_name

  # Give the Karpenter controller permission via a role it assumes (IRSA):
  # "IAM Roles for Service Accounts" — the K8s service account Karpenter runs as
  # is mapped to this IAM role, so the controller gets AWS permissions without
  # stored keys (same temporary-credential idea as your SSO).
  enable_v1_permissions = true

  # The namespace + service account Karpenter's controller will run as.
  namespace = "kube-system"

  # Nodes Karpenter launches will wear this role; attach the SSM policy so they
  # can be managed/debugged via Session Manager (like your RDS tunnel instance).
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }

  tags = {
    Name = "${var.cluster_name}-karpenter"
  }
}

# --- Karpenter controller (installed via Helm into the cluster) ---
# The controller software that watches for pending pods and provisions nodes.
# Runs in kube-system, using the IRSA role from the karpenter sub-module.
resource "helm_release" "karpenter" {
  namespace  = "kube-system"
  name       = "karpenter"
  repository = "oci://public.ecr.aws/karpenter"
  chart      = "karpenter"
  version    = "1.2.1"

  values = [yamlencode({
    serviceAccount = {
      annotations = {
        "eks.amazonaws.com/role-arn" = module.karpenter.iam_role_arn
      }
    }
    settings = {
      clusterName       = module.eks.cluster_name
      interruptionQueue = module.karpenter.queue_name
    }
  })]
}

# --- EC2NodeClass: the AWS-specific template for nodes Karpenter launches ---
# Says which AMI family, which IAM role the nodes wear, and how Karpenter
# discovers the subnets/security groups to use (via the karpenter.sh/discovery
# tag we set on the cluster + VPC). This is the "how to build each node" recipe.
resource "kubernetes_manifest" "karpenter_node_class" {
  manifest = {
    apiVersion = "karpenter.k8s.aws/v1"
    kind       = "EC2NodeClass"
    metadata   = { name = "default" }
    spec = {
      amiFamily = "AL2023"
      amiSelectorTerms = [{
        alias = "al2023@latest"
      }]
      role = module.karpenter.node_iam_role_name
      subnetSelectorTerms = [{
        tags = { "karpenter.sh/discovery" = var.cluster_name }
      }]
      securityGroupSelectorTerms = [{
        tags = { "karpenter.sh/discovery" = var.cluster_name }
      }]
    }
  }

  depends_on = [helm_release.karpenter]
}

# --- NodePool: the provisioning rules — what Karpenter MAY launch ---
# Limits Karpenter to sensible instance types, on-demand, references the node
# class above. This is the "menu of nodes the restaurant may add."
resource "kubernetes_manifest" "karpenter_node_pool" {
  manifest = {
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata   = { name = "default" }
    spec = {
      template = {
        spec = {
          nodeClassRef = {
            group = "karpenter.k8s.aws"
            kind  = "EC2NodeClass"
            name  = "default"
          }
          requirements = [
            {
              key      = "karpenter.sh/capacity-type"
              operator = "In"
              values   = ["on-demand"]
            },
            {
              key      = "kubernetes.io/arch"
              operator = "In"
              values   = ["amd64"]
            }
          ]
        }
      }
      limits = {
        cpu = "100"
      }
      disruption = {
        consolidationPolicy = "WhenEmptyOrUnderutilized"
        consolidateAfter    = "30s"
      }
    }
  }

  depends_on = [helm_release.karpenter]
}