library(data.table)
library(sf)
library(geojsonsf)

# --- Einstellungen ---
input_file  <- "D:/enviprojects/Berlin_Mehringdamm_adatrace_g6/mehringdamm_adatrace_g12_all.csv" 
output_file <- "D:/enviprojects/Berlin_Mehringdamm_adatrace_g6/mehringdamm_adatrace_g12.gpkg"

chunk_size  <- 20000 
base_time   <- as.POSIXct("2024-01-01 00:00:00", tz="UTC")

# Alte Datei löschen, falls vorhanden
if (file.exists(output_file)) {
  file.remove(output_file)
  cat("Bestehendes GeoPackage gelöscht.\n")
}

# --- Hilfsfunktion für den Hex-WKB-Zweig ---
hex_to_sfc <- function(hex_strings) {
  raw_list <- lapply(hex_strings, function(x) {
    if (is.na(x) || x == "") return(NULL)
    as.raw(as.hexmode(substring(x, seq(1, nchar(x)-1, 2), seq(2, nchar(x), 2))))
  })
  st_as_sfc(raw_list, EWKB = TRUE)
}

# --- Format-Analyse (Auto-Detect) ---
cat("Untersuche Dateiformat...\n")
header <- names(fread(input_file, nrows = 0))

if ("trip_object" %in% header) {
  format_type <- "json"
  cat("Format erkannt: Verschachteltes JSON (trip_object).\n")
} else if (any(c("geom", "geometry") %in% header)) {
  format_type <- "hex"
  cat("Format erkannt: Hex-String/WKB (geom/geometry).\n")
} else {
  stop("Abbruch: Unbekanntes Dateiformat. Keine 'trip_object', 'geom' oder 'geometry' Spalte gefunden.")
}

offset <- 0
chunk_id <- 1

# =====================================================================
# ZWEIG 1: HEX-STRING / WKB VERARBEITUNG
# =====================================================================
if (format_type == "hex") {
  
  # Spalten ermitteln
  id_col_name <- if("trip_id" %in% header) "trip_id" else if("vehicle_id" %in% header) "vehicle_id" else header[1]
  geom_col_name <- if("geom" %in% header) "geom" else "geometry"
  
  cols_to_keep <- c(id_col_name, geom_col_name)
  if ("start_timestamp" %in% header) cols_to_keep <- c(cols_to_keep, "start_timestamp")
  
  cat("Starte chunk-weise Verarbeitung (Hex/WKB)...\n")
  
  repeat {
    dt_chunk <- tryCatch({
      if (chunk_id == 1) {
        fread(input_file, nrows = chunk_size, select = cols_to_keep, fill = TRUE)
      } else {
        temp <- fread(input_file, nrows = chunk_size, skip = offset + 1, header = FALSE, fill = TRUE)
        setnames(temp, header)
        temp[, ..cols_to_keep]
      }
    }, error = function(e) { NULL })
    
    if (is.null(dt_chunk) || nrow(dt_chunk) == 0) {
      cat("Ende der Datei erreicht.\n")
      break
    }
    
    # Leere Geometrien entfernen
    dt_chunk <- dt_chunk[!is.na(get(geom_col_name)) & get(geom_col_name) != ""]
    
    if (nrow(dt_chunk) > 0) {
      geo_col_data <- dt_chunk[[geom_col_name]]
      geometry_sfc <- hex_to_sfc(geo_col_data)
      
      # Alte Text-Spalte löschen
      dt_chunk[, (geom_col_name) := NULL]
      
      # SF Objekt erstellen
      sf_chunk <- st_sf(dt_chunk, geometry = geometry_sfc)
      
      if ("start_timestamp" %in% names(sf_chunk)) {
        sf_chunk$start_timestamp <- as.numeric(sf_chunk$start_timestamp)
        sf_chunk$time_start <- format(base_time + sf_chunk$start_timestamp, "%H:%M:%S")
      }
      
      sf_chunk$track_id <- sf_chunk[[id_col_name]]
      
      if (is.na(st_crs(sf_chunk))) st_crs(sf_chunk) <- 4326 
      
      st_write(sf_chunk, output_file, append = (chunk_id > 1), quiet = TRUE)
      cat(sprintf("  Chunk %d verarbeitet (%d Zeilen)...\n", chunk_id, nrow(sf_chunk)))
    }
    
    offset <- offset + chunk_size
    chunk_id <- chunk_id + 1
  }
  
  # =====================================================================
  # ZWEIG 2: JSON / GEOJSON VERARBEITUNG
  # =====================================================================
} else if (format_type == "json") {
  
  # Spalten für JSON ermitteln (wir prüfen ob group_id existiert, sonst Fallback)
  id_col_name <- if("group_id" %in% header) "group_id" else header[1]
  cols_to_keep <- c(id_col_name, "trip_object")
  
  cat("Starte chunk-weise Verarbeitung (JSON)...\n")
  
  repeat {
    dt_chunk <- tryCatch({
      if (chunk_id == 1) {
        fread(input_file, nrows = chunk_size, select = cols_to_keep, fill = TRUE)
      } else {
        temp <- fread(input_file, nrows = chunk_size, skip = offset + 1, header = FALSE, fill = TRUE)
        setnames(temp, header)
        temp[, ..cols_to_keep]
      }
    }, error = function(e) { NULL })
    
    if (is.null(dt_chunk) || nrow(dt_chunk) == 0) {
      cat("Ende der Datei erreicht.\n")
      break
    }
    
    dt_chunk <- dt_chunk[!is.na(trip_object) & trip_object != ""]
    
    if (nrow(dt_chunk) > 0) {
      # Fix für doppelte Anführungszeichen
      if (grepl('""', substr(dt_chunk$trip_object[1], 1, 100), fixed = TRUE)) {
        dt_chunk[, trip_object := gsub('""', '"', trip_object, fixed = TRUE)]
        dt_chunk[, trip_object := gsub('^"|"$', '', trip_object)]
      }
      
      # JSON Parsen
      sf_chunk <- tryCatch({
        geojson_sf(dt_chunk$trip_object)
      }, error = function(e) {
        cat("  Standard-Parsing fehlgeschlagen. Versuche Zeile-für-Zeile Fallback...\n")
        valid_rows <- list()
        for (i in seq_len(nrow(dt_chunk))) {
          try({
            item <- geojson_sf(dt_chunk$trip_object[i])
            item[[id_col_name]] <- dt_chunk[[id_col_name]][i]
            valid_rows[[length(valid_rows)+1]] <- item
          }, silent = TRUE)
        }
        return(do.call(rbind, valid_rows))
      })
      
      # Zeitstempel extrahieren
      if (!is.null(sf_chunk) && nrow(sf_chunk) > 0) {
        if (!id_col_name %in% names(sf_chunk) && nrow(sf_chunk) == nrow(dt_chunk)) {
          sf_chunk[[id_col_name]] <- dt_chunk[[id_col_name]]
        }
        
        coords <- as.data.table(st_coordinates(sf_chunk))
        
        if ("M" %in% names(coords)) {
          times <- coords[, .(start_sec = head(M, 1), end_sec = tail(M, 1)), by = L1]
          
          sf_chunk$seconds_start <- times$start_sec
          sf_chunk$seconds_end   <- times$end_sec
          sf_chunk$time_start <- format(base_time + sf_chunk$seconds_start, "%H:%M:%S")
          sf_chunk$time_end   <- format(base_time + sf_chunk$seconds_end, "%H:%M:%S")
        } else {
          # Wird nur einmal pro Chunk geworfen, falls keine M-Werte existieren
          if (chunk_id == 1) cat("  Warnung: Keine M-Werte (Zeitstempel) in den Koordinaten gefunden.\n")
        }
        
        st_write(sf_chunk, output_file, driver = "GPKG", append = (chunk_id > 1), quiet = TRUE)
        cat(sprintf("  Chunk %d verarbeitet (%d Zeilen)...\n", chunk_id, nrow(sf_chunk)))
      }
    }
    
    offset <- offset + nrow(dt_chunk) # Wichtig: nrow(dt_chunk) statt chunk_size, falls der letzte Chunk kleiner ist
    chunk_id <- chunk_id + 1
    
    rm(dt_chunk, sf_chunk)
    gc()
  }
}

cat("Fertig!\n")