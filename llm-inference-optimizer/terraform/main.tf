# Minimal stub for provisioning a single GPU node pool on a cloud provider.
# This is intentionally provider-agnostic scaffolding — fill in the provider
# block matching where you actually deploy (AWS/GCP/Azure/Lambda Labs, etc).

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {}
variable "region" { default = "us-central1" }
variable "gpu_type" { default = "nvidia-tesla-t4" }
variable "gpu_count" { default = 1 }
variable "machine_type" { default = "n1-standard-8" }

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "inference_cluster" {
  name     = "llm-inference-cluster"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "gpu_pool" {
  name       = "gpu-pool"
  cluster    = google_container_cluster.inference_cluster.name
  location   = var.region
  node_count = 1

  node_config {
    machine_type = var.machine_type
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count
    }
    oauth_scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}

output "cluster_endpoint" {
  value = google_container_cluster.inference_cluster.endpoint
}
