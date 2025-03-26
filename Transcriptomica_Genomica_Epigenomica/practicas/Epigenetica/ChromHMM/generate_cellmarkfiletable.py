import os

# Lista de tus archivos .bed.gz de marcas
files = [
    "wgEncodeBroadHistoneMonocd14ro1746CtcfAlnRep1_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746CtcfAlnRep2_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K04me1AlnRep1_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K04me1AlnRep2_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K04me3AlnRep1_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K04me3AlnRep2_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K09me3AlnRep1_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K09me3AlnRep2_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K27acAlnRep1_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K27acAlnRep2_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K27me3AlnRep1_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K27me3AlnRep2_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K36me3AlnRep1_RM_chr16.bed.gz",
    "wgEncodeBroadHistoneMonocd14ro1746H3K36me3AlnRep2_RM_chr16.bed.gz"
]

# Asumimos que los archivos de control siguen una convención similar a las marcas.
# Podemos mapear la marca con el archivo de control correspondiente.
control_files = {
    "Ctcf": [
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep1_RM_chr16.bed.gz",
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep2_RM_chr16.bed.gz"
    ],
    "H3K04me1": [
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep1_RM_chr16.bed.gz",
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep2_RM_chr16.bed.gz"
    ],
    "H3K04me3": [
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep1_RM_chr16.bed.gz",
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep2_RM_chr16.bed.gz"
    ],
    "H3K09me3": [
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep1_RM_chr16.bed.gz",
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep2_RM_chr16.bed.gz"
    ],
    "H3K27ac": [
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep1_RM_chr16.bed.gz",
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep2_RM_chr16.bed.gz"
    ],
    "H3K27me3": [
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep1_RM_chr16.bed.gz",
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep2_RM_chr16.bed.gz"
    ],
    "H3K36me3": [
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep1_RM_chr16.bed.gz",
        "wgEncodeBroadHistoneMonocd14ro1746ControlAlnRep2_RM_chr16.bed.gz"
    ]
}

# Crear el archivo de salida
with open('config_chromHmm_bin.txt', 'w') as outfile:
    # Escribir encabezado
    outfile.write("cell_type\tmark\tfile\tcontrol_file\n")
    
    # Generar líneas para cada archivo de marca
    for file in files:
        # Extraer información del archivo
        parts = file.split('wgEncodeBroadHistoneMonocd14ro1746')
        mark = parts[1].split('Aln')[0]  # Esto extrae la marca (ej. "Ctcf", "H3K04me1")
        
        # Determinar el tipo de célula (usaremos el prefijo de cada archivo como tipo de célula)
        if "Ctcf" in mark:
            cell_type = "cell1"
        elif "H3K04me1" in mark or "H3K04me3" in mark:
            cell_type = "cell2"
        elif "H3K09me3" in mark or "H3K27ac" in mark:
            cell_type = "cell3"
        elif "H3K27me3" in mark or "H3K36me3" in mark:
            cell_type = "cell4"
        else:
            cell_type = "unknown"
        
        # Asignar el archivo de control correspondiente
        control_file = control_files.get(mark, ["no_control"])[0]  # Si no hay control, asigna "no_control"
        
        # Crear la línea con el formato adecuado
        line = f"{cell_type}\t{mark}\t{file}\t{control_file}\n"
        
        # Escribir la línea en el archivo
        outfile.write(line)

print("Archivo config_chromHmm_bin.txt generado con éxito.")
