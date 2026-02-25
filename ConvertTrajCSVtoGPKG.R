library(data.table)
library(sf)

# --- Einstellungen ---
# Pfad zur neuen Datei anpassen (z.B. NewExampleFile1.txt oder NewExampleFile2.txt)
input_file  <- "D:/enviprojects/Berlin_Feinstaub/traj/k7_mehringdamm.csv" 
output_file <- "D:/enviprojects/Berlin_Feinstaub/traj/k7_mehringdamm.gpkg"

chunk_size  <- 20000 
# Basis-Zeit (falls nötig)
base_time <- as.POSIXct("2024-01-01 00:00:00", tz="UTC")

# Alte Datei löschen, falls vorhanden
if (file.exists(output_file)) file.remove(output_file)

offset <- 0
chunk_id <- 1

# --- Hilfsfunktion: Hex-String zu sfc ---
hex_to_sfc <- function(hex_strings) {
  raw_list <- lapply(hex_strings, function(x) {
    if (is.na(x) || x == "") return(NULL)
    as.raw(as.hexmode(substring(x, seq(1, nchar(x)-1, 2), seq(2, nchar(x), 2))))
  })
  st_as_sfc(raw_list, EWKB = TRUE)
}

cat("Starte Verarbeitung (mit Spaltenbereinigung)...\n")

repeat {
  # --- 1. Chunk lesen ---
  dt_chunk <- tryCatch({
    if (chunk_id == 1) {
      # Header analysieren
      header <- names(fread(input_file, nrows = 0))
      
      id_col <- if("trip_id" %in% header) "trip_id" else if("vehicle_id" %in% header) "vehicle_id" else header[1]
      geom_col <- if("geom" %in% header) "geom" else "geometry"
      time_col <- "start_timestamp"
      
      cols_to_keep <- c(id_col, geom_col)
      if (time_col %in% header) cols_to_keep <- c(cols_to_keep, time_col)
      
      # Globale Variablen für Folgeschleifen setzen
      assign("cols_to_keep", cols_to_keep, envir = .GlobalEnv)
      assign("id_col_name", id_col, envir = .GlobalEnv)
      assign("geom_col_name", geom_col, envir = .GlobalEnv) # Speichern, wie die Spalte heißt
      
      fread(input_file, nrows = chunk_size, select = cols_to_keep, fill = TRUE)
    } else {
      # Folge-Chunks lesen
      temp_chunk <- fread(input_file, nrows = chunk_size, skip = offset + 1, header = FALSE, fill = TRUE)
      full_header <- names(fread(input_file, nrows = 0))
      setnames(temp_chunk, full_header)
      temp_chunk[, ..cols_to_keep]
    }
  }, error = function(e) { NULL })
  
  if (is.null(dt_chunk) || nrow(dt_chunk) == 0) {
    cat("Ende der Datei erreicht.\n")
    break
  }
  
  # Leere Geometrien entfernen
  dt_chunk <- dt_chunk[!is.na(get(geom_col_name)) & get(geom_col_name) != ""]
  
  if (nrow(dt_chunk) > 0) {
    
    # --- 2. Geometrie umwandeln ---
    geo_col_data <- dt_chunk[[geom_col_name]]
    geometry_sfc <- hex_to_sfc(geo_col_data)
    
    # --- WICHTIGE ÄNDERUNG: Alte Text-Spalte löschen ---
    # Wir entfernen die Spalte 'geom' aus den Daten, damit sie nicht mit der neuen 
    # Geometrie-Spalte des GeoPackages kollidiert.
    dt_chunk[, (geom_col_name) := NULL]
    
    # --- 3. SF Objekt erstellen ---
    sf_chunk <- st_sf(dt_chunk, geometry = geometry_sfc)
    
    # 4. Zeitstempel & CRS (wie gehabt)
    if ("start_timestamp" %in% names(sf_chunk)) {
      sf_chunk$start_timestamp <- as.numeric(sf_chunk$start_timestamp)
      sf_chunk$time_start <- format(base_time + sf_chunk$start_timestamp, "%H:%M:%S")
    }
    
    sf_chunk$track_id <- sf_chunk[[id_col_name]]
    
    if (is.na(st_crs(sf_chunk))) {
      st_crs(sf_chunk) <- 4326 
    }
    
    # 5. Schreiben
    st_write(sf_chunk, output_file, append = (chunk_id > 1), quiet = TRUE)
    
    cat(sprintf("Chunk %d verarbeitet (%d Zeilen)...\n", chunk_id, nrow(sf_chunk)))
  }
  
  offset <- offset + chunk_size
  chunk_id <- chunk_id + 1
}

cat("Fertig!\n")