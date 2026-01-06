library(data.table)
library(sf)
library(geojsonsf)

# --- Einstellungen ---
input_file  <- "D:/ber_linestrings_subarea_friedrichstr_gs_1.csv"
output_file <- "D:/ber_linestrings_cleaned.gpkg"
chunk_size  <- 20000 

# Alte Datei löschen, falls vorhanden
if (file.exists(output_file)) file.remove(output_file)

offset <- 0
chunk_id <- 1

cat("Starte robuste Verarbeitung...\n")

repeat {
  # 1. Chunk lesen (mit Fehler-Handling für Dateiende)
  dt_chunk <- tryCatch({
    if (chunk_id == 1) {
      fread(input_file, nrows = chunk_size, select = c("group_id", "trip_object"), fill = TRUE)
    } else {
      fread(input_file, nrows = chunk_size, skip = offset + 1, header = FALSE, fill = TRUE, 
            select = c(1, 2), col.names = c("group_id", "trip_object"))
    }
  }, error = function(e) { return(NULL) })
  
  if (is.null(dt_chunk) || nrow(dt_chunk) == 0) {
    cat("Ende der Datei erreicht.\n")
    break
  }
  
  cat(sprintf("Verarbeite Chunk %d (%d Zeilen)...\n", chunk_id, nrow(dt_chunk)))
  
  # 2. Bereinigen
  dt_chunk <- dt_chunk[!is.na(trip_object) & trip_object != ""]
  
  if (nrow(dt_chunk) > 0) {
    # Fix für doppelte Anführungszeichen (falls nötig)
    if (grepl('""', substr(dt_chunk$trip_object[1], 1, 100), fixed = TRUE)) {
      dt_chunk[, trip_object := gsub('""', '"', trip_object, fixed = TRUE)]
      dt_chunk[, trip_object := gsub('^"|"$', '', trip_object)]
    }
    
    # 3. JSON Parsen
    sf_chunk <- tryCatch({
      geojson_sf(dt_chunk$trip_object)
    }, error = function(e) {
      cat("  Standard-Parsing fehlgeschlagen. Versuche Zeile-für-Zeile Fallback...\n")
      valid_rows <- list()
      for (i in 1:nrow(dt_chunk)) {
        try({
          item <- geojson_sf(dt_chunk$trip_object[i])
          item$group_id <- dt_chunk$group_id[i]
          valid_rows[[length(valid_rows)+1]] <- item
        }, silent = TRUE)
      }
      return(do.call(rbind, valid_rows))
    })
    
    # 4. Zeitstempel extrahieren (NEUE METHODE OHNE st_startpoint)
    if (!is.null(sf_chunk) && nrow(sf_chunk) > 0) {
      
      # IDs retten
      if (!"group_id" %in% names(sf_chunk) && nrow(sf_chunk) == nrow(dt_chunk)) {
        sf_chunk$group_id <- dt_chunk$group_id
      }
      
      # Wir holen alle Koordinaten als Matrix (X, Y, Z, M, L1)
      # L1 ist die ID der Linie, M ist der Zeitstempel
      coords <- as.data.table(st_coordinates(sf_chunk))
      
      # Wir nehmen an, dass die 4. Spalte 'M' ist. 
      # Falls die Spaltennamen fehlen, ist M oft Spalte 4 (Z ist 3).
      # st_coordinates gibt normalerweise Spaltennamen aus (X, Y, Z, M).
      
      if ("M" %in% names(coords)) {
        # Gruppieren nach 'L1' (Linien-ID) und ersten/letzten Wert holen
        times <- coords[, .(
          start_sec = head(M, 1), 
          end_sec = tail(M, 1)
        ), by = L1]
        
        sf_chunk$seconds_start <- times$start_sec
        sf_chunk$seconds_end   <- times$end_sec
        
        # In lesbare Zeit umwandeln
        base_time <- as.POSIXct("2024-01-01 00:00:00", tz="UTC")
        sf_chunk$time_start <- format(base_time + sf_chunk$seconds_start, "%H:%M:%S")
        sf_chunk$time_end   <- format(base_time + sf_chunk$seconds_end, "%H:%M:%S")
      } else {
        cat("  Warnung: Keine M-Werte (Zeitstempel) in den Koordinaten gefunden.\n")
      }
      
      # 5. Speichern (Anhängen)
      st_write(sf_chunk, output_file, driver = "GPKG", append = TRUE, quiet = TRUE)
    }
  }
  
  offset <- offset + nrow(dt_chunk)
  chunk_id <- chunk_id + 1
  
  rm(dt_chunk, sf_chunk)
  gc()
}

cat("Fertig!")