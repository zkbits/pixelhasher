#include <stdlib.h>
#include <stdio.h>

void gpu_init();

void gpu_init(char *gpuDescription)
{
  cudaDeviceProp device_prop;
  int device_count;

  cudaGetDeviceCount(&device_count);
  if (device_count != 1) {
    printf("\nHelp: Select a GPU like this:\n\n");
    printf("  CUDA_VISIBLE_DEVICES=n ./pixelhasher\n");
    printf("                       ^ put GPU index here (0 is first GPU)\n\n");
    exit(EXIT_FAILURE);
  }

  if (cudaGetDeviceProperties(&device_prop, 0) != cudaSuccess) {
    printf("Problem getting properties for device, exiting...\n");
    exit(EXIT_FAILURE);
  } 

  printf("name:        %s\n", device_prop.name);
  printf("pciBusID:    %d\n", device_prop.pciBusID);
  printf("pciDeviceID: %d\n", device_prop.pciDeviceID);
  // todo in future: calculate more optimal (threads x blocks) from device_prop? micro speedup only

  sprintf(gpuDescription, "%s, PCI:%d:%d", device_prop.name, device_prop.pciBusID, device_prop.pciDeviceID);
}

int main(int argc, char **argv)
{
  char gpuDescription[128];
  gpu_init(gpuDescription);
  printf("%s\n", gpuDescription);
}
