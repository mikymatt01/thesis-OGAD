terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = local.region
  zone    = var.zone
}

locals {
  # Zone is the only thing your loop varies; region is derived automatically.
  # Example: "europe-west4-a" -> "europe-west4"
  region = replace(var.zone, "-[a-z]$", "")

  # A2/A3 accelerator-optimized machine types already include GPUs as part of the
  # machine type. In that case, do not also specify guest_accelerators.
  accelerator_optimized_prefixes = [
    "a2-",
    "a3-",
  ]
  is_accelerator_optimized = anytrue([
    for p in local.accelerator_optimized_prefixes : startswith(var.machine_type, p)
  ])
}

resource "google_compute_instance" "thesis_gpu_vm" {
  name         = var.instance_name
  zone         = var.zone
  machine_type = var.machine_type

  tags = var.network_tags

  boot_disk {
    auto_delete = true
    initialize_params {
      image = var.boot_image
      size  = var.boot_disk_gb
      type  = var.boot_disk_type
    }
  }

  network_interface {
    network    = var.network
    subnetwork = var.subnetwork

    access_config {}
  }

  dynamic "guest_accelerator" {
    for_each = local.is_accelerator_optimized ? [] : [1]
    content {
      type  = var.gpu_type
      count = var.gpu_count
    }
  }

  # Required for GPU instances
  scheduling {
    on_host_maintenance = "TERMINATE"
    # Spot/preemptible VMs cannot use automatic restart.
    automatic_restart  = var.preemptible ? false : true
    preemptible        = var.preemptible
    provisioning_model = var.preemptible ? "SPOT" : "STANDARD"
  }

  metadata = {
    enable-oslogin = var.enable_oslogin ? "TRUE" : "FALSE"
  }

  service_account {
    email  = var.service_account_email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}

output "instance_self_link" {
  value = google_compute_instance.thesis_gpu_vm.self_link
}

output "instance_external_ip" {
  value = google_compute_instance.thesis_gpu_vm.network_interface[0].access_config[0].nat_ip
}

variable "project_id" {
  type        = string
  description = "GCP project ID."
  default     = "tesi-383415"
}

variable "region" {
  type        = string
  description = "DEPRECATED: region is derived from var.zone; keep only for backward compatibility."
  default     = null
  nullable    = true
}

variable "zone" {
  type        = string
  description = "GCP zone where the instance will be created."
  default     = "us-central1-a"
}

variable "instance_name" {
  type        = string
  description = "Compute Engine instance name."
  default     = "ogad-thesis-t4-n8"
}

variable "machine_type" {
  type        = string
  description = "Machine type. For the common 'N8' shape on N1 + attached T4, use n1-standard-8."
  default     = "n1-standard-8"
}

variable "gpu_type" {
  type        = string
  description = "GPU accelerator type for non-accelerator-optimized machine types (attached GPUs). For T4 use nvidia-tesla-t4."
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  type        = number
  description = "Number of GPUs."
  default     = 1
}

variable "network" {
  type        = string
  description = "VPC network name or self_link."
  default     = "default"
}

variable "subnetwork" {
  type        = string
  description = "Subnetwork name or self_link. Leave null to use the network's default subnetwork."
  default     = null
}

variable "network_tags" {
  type        = list(string)
  description = "Network tags (e.g., allow-ssh)."
  default     = ["allow-ssh"]
}

variable "boot_image" {
  type        = string
  description = "Boot disk image (family or specific image). Deep Learning VM on Linux (GPU-ready) by default."
  # This matches the Compute Engine UI "Deep Learning on Linux" images (Deep Learning VM).
  # Change to e.g. Debian/Ubuntu if you prefer a plain OS.
  # "Deep Learning VM with CUDA M132" (Ubuntu 22.04, CUDA 12.9 / cu129).
  default = "projects/deeplearning-platform-release/global/images/family/common-cu129-ubuntu-2204-nvidia-580"
}

variable "boot_disk_gb" {
  type        = number
  description = "Boot disk size in GB."
  default     = 100
}

variable "boot_disk_type" {
  type        = string
  description = "Boot disk type."
  default     = "pd-balanced"
}

variable "preemptible" {
  type        = bool
  description = "If true, creates a Spot VM (cheaper but can be stopped)."
  default     = false
}

variable "enable_oslogin" {
  type        = bool
  description = "Enable OS Login."
  default     = true
}

variable "service_account_email" {
  type        = string
  description = "Service account email for the VM. Use 'default' to attach the project's default compute SA."
  default     = "default"
}
