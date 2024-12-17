################################################################################
# BIAS: Bias of estimated β’s as percentage of the true β
# Graph generation
################################################################################


rm(list=ls())
# Libraries
library(tidyverse)
library(ggplot2)
library(readr)
library(stringr)
library(purrr)
library(scales)

# Data graphing function
graficas_box_plot <- function(subfolder) {
  

  # Data preparation
  folder_base <- "C:/Projects/ProductAvaility/Repository"
  folder_orig <- file.path(folder_base, subfolder)
  files_dir_list <- list.files(folder_orig, pattern = "^J\\d", full.names = TRUE)
  
  for (path_dir in files_dir_list){
    folder_path <- file.path(path_dir, 'dataframes')
    base_path <- dirname(folder_path)  
    graph_folder <- file.path(base_path, 'boxplot') 
    
    if (!dir.exists(graph_folder)) {
      dir.create(graph_folder)
    }
    
    true_value_beta1 <- 2
    all_data <- list()
    titulo_grafico <- ""
    
    csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
    
    for (file_path in csv_files) {
      df <- read_csv(file_path)
      
      if ("beta1" %in% names(df) && "model" %in% names(df)) {
        
        df <- df %>% filter(success == TRUE) %>%
          mutate(
                 sesgo = ((beta1 - true_value_beta1) / true_value_beta1) * 100 # Relative bias
                 )
    
        match <- str_extract(basename(file_path), "q-\\d+")
        df$archivo <- ifelse(!is.na(match), str_replace(match, "q-", "q="), basename(file_path))

        all_data[[length(all_data) + 1]] <- df %>% select(archivo, model, sesgo)
        
        J_filename <- as.integer(str_extract(file_path, "(?<=/J)\\d+"))
      }
    }
    
    combined_df <- bind_rows(all_data)
    
    combined_df <- combined_df %>%
      mutate(model = recode(model, 
                            "Sequential" = "Capacity-constrained", 
                            "No Sequential" = "Standard"))

    combined_df <- combined_df %>%
      mutate(valor_q = as.integer(str_extract(archivo, "\\d+"))) %>%
      arrange(valor_q)
    
    n_observaciones <- combined_df %>%
      group_by(valor_q, model) %>%
      summarise(
        n = n(),                          
        .groups = 'drop'
      )
    
    combined_df <- combined_df %>%
      mutate(num_digits = nchar(str_extract(archivo, "\\d+"))) %>% 
      arrange(desc(num_digits), desc(archivo)) 
      
    stats_df <- combined_df %>%
      group_by(valor_q, model) %>%
      summarise(
        min = min(sesgo, na.rm = TRUE),
        Q1 = quantile(sesgo, 0.25, na.rm = TRUE),
        median = median(sesgo, na.rm = TRUE),
        Q3 = quantile(sesgo, 0.75, na.rm = TRUE),
        max = max(sesgo, na.rm = TRUE),
        IQR = IQR(sesgo, na.rm = TRUE),
        lower_bound = Q1 - 1.5 * IQR,
        upper_bound = Q3 + 1.5 * IQR
      )

    lower_limit <- -275
    upper_limit <- 275
    
    palette_lines <- c("#ee5f2e","darkblue") 
    palette_fill <- muted(palette_lines, l = 80) 
    
    combined_df$model <- factor(combined_df$model, levels = c("Capacity-constrained","Standard"))

    # Plot
    p <- ggplot(combined_df, aes(y = factor(valor_q, levels = unique(valor_q)), 
                                 x = sesgo, fill = model, colour = model)) +
      geom_boxplot(width = 0.5, outlier.shape = NA, size = 0.5) +
      scale_color_manual(values = palette_lines, name = "Model", 
                         breaks = c("Standard","Capacity-constrained"),
                         labels = c("Capacity-constrained" = "Capacity-\nconstrained", 
                                    "Standard" = "Standard\n ")) +
      scale_fill_manual(values = palette_fill,
                        breaks = c("Standard","Capacity-constrained"),
                        guide = "none") + 
      labs(title = "", y = "Capacity(q)", x = "Bias (%)") +
      theme_minimal(base_size = 15) +
      theme(
        axis.title.x = element_text(size = 28),
        axis.title.y = element_text(size = 28),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 24),
        axis.text.y = element_text(angle = 0, size = 24),
        legend.title = element_text(size = 24),
        legend.text = element_text(size = 20),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.ticks = element_line(colour = "black")
      ) +
      geom_vline(xintercept = 0, color = "gray20", linetype = "dotted", size = 1) +
      scale_x_continuous(
        limits = c(lower_limit, upper_limit),
        breaks = seq(floor(lower_limit/50)*50, ceiling(upper_limit/50)*50, by = 50)
      ) +
      geom_text(data = n_observaciones, aes(label = paste("S=", n,sep=""), x = upper_limit - 25), 
                position = position_dodge(width = 0.5), 
                vjust = -0.0, size = 5.5, 
                color = "black", angle = 0) + 
      guides(color  = guide_legend(override.aes = list(size = 15))) + 
      guides(color = guide_legend(override.aes = list(fill = palette_fill, size = 0)))
    
    output_path <- file.path(graph_folder, "boxplot_relative_bias.pdf")
    ggsave(output_path, plot = p, width = 30, height = 32, units = "cm") 
    
    # Bias stats
    mediana_sesgo_tabla <- combined_df %>%
      group_by(archivo, model) %>%
      summarize(`Median bias` = median(sesgo, na.rm = TRUE)) %>%
      filter(model == "Standard") %>%
      rename(q = archivo)
    
    mediana_sesgo_tabla$q <- as.numeric(gsub("\\D", "", mediana_sesgo_tabla$q))
    
    mediana_sesgo_tabla<- mediana_sesgo_tabla %>%
      arrange(q) %>%
      select(-model)
    
    output_stats_path <- file.path(graph_folder, "relative_bias_stats.csv")
    write.csv(mediana_sesgo_tabla, output_stats_path, row.names = FALSE)
    
  }
}

# Run function

graficas_box_plot("homog")
graficas_box_plot("heterog")
