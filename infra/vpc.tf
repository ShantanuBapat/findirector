# infra/vpc.tf
#
# WHAT THIS CREATES:
#   The whole network foundation, via the official AWS VPC module. From ~20 lines
#   of settings, the module produces: the VPC, 2 public + 2 private subnets across
#   2 AZs, an internet gateway, ONE NAT gateway (+ its Elastic IP), and the route
#   tables wiring it all together (public subnets -> IGW, private subnets -> NAT).
#
# The subnet CIDRs match the approved IP plan (docs/architecture.pdf):
#   private /20 (4,096 IPs each — EKS assigns an IP per pod, so sized large)
#   public  /24 (256 IPs each — only the ALB + NAT live here)

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  # The VPC itself and its address range.
  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  # Two availability zones (physically separate data centers) for resilience —
  # and because RDS requires subnets in >= 2 AZs.
  azs = ["${var.aws_region}a", "${var.aws_region}b"]

  # PRIVATE subnets (one per AZ): RDS, EKS pods, vLLM live here — hidden from the
  # internet. Route table sends 0.0.0.0/0 -> NAT. Sized /20 for EKS pod IPs.
  private_subnets = ["10.0.0.0/20", "10.0.16.0/20"]

  # PUBLIC subnets (one per AZ): load balancer + NAT gateway — internet-facing.
  # Route table sends 0.0.0.0/0 -> internet gateway. Small /24 is plenty.
  public_subnets = ["10.0.32.0/24", "10.0.33.0/24"]

  # NAT gateway: enabled, and a SINGLE shared NAT (cost-conscious — ~$32/mo for
  # one vs ~2x for one-per-AZ). The one-way outbound valve for private subnets.
  enable_nat_gateway = true
  single_nat_gateway = true

  # Internal DNS names for resources (so services find each other by hostname).
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tag subnets by tier, so it's obvious which is which in the console.
  public_subnet_tags  = { Tier = "public" }
  private_subnet_tags = { Tier = "private" }
}