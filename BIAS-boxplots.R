################################################################################
# BETA: Estimated β’s distribution and statistics
# Graph generation + summary table construction
################################################################################

rm(list = ls())

# Load required libraries
library(tidyverse)
library(ggplot2)
library(readr)
library(stringr)
library(purrr)
library(scales)
library(tibble)
library(readr) 

# Define base folder
folder_base <- "C:/Projects/ProductAvaility/Repository"

# Function to generate boxplots and beta statistics
graficas_box_plot <- function(subfolder) {
  folder_orig <- file.path(folder_base, subfolder)
  files_dir_list <- list.files(folder_orig, pattern = "^J\\d", full.names = TRUE)
  
  for (path_dir in files_dir_list) {
    folder_path <- file.path(path_dir, 'dataframes')
    graph_folder <- file.path(path_dir, 'boxplot') 
    if (!dir.exists(graph_folder)) dir.create(graph_folder)
    
    true_value_beta1 <- 2
    all_data <- list()
    
    csv_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
    for (file_path in csv_files) {
      df <- read_csv(file_path, show_col_types = FALSE)
      if ("beta1" %in% names(df) && "model" %in% names(df)) {
        df <- df %>% 
          filter(success == TRUE) %>%
          mutate(beta_predicted = beta1)
        
        match <- str_extract(basename(file_path), "q-\\d+")
        df$archivo <- ifelse(!is.na(match), str_replace(match, "q-", "q="), basename(file_path))
        all_data[[length(all_data) + 1]] <- df %>% select(archivo, model, beta_predicted)
      }
    }
    
    combined_df <- bind_rows(all_data) %>%
      mutate(model = recode(model, "Sequential" = "Capacity-constrained", "No Sequential" = "Standard"),
             valor_q = as.integer(str_extract(archivo, "\\d+")))
    
    combined_df$model <- factor(combined_df$model, levels = c("Capacity-constrained", "Standard"))
    
    # Save beta stats for Standard model
    beta_stats <- combined_df %>%
      filter(model == "Standard") %>%
      group_by(archivo) %>%
      summarise(
        Mean = mean(beta_predicted, na.rm = TRUE),
        SD = sd(beta_predicted, na.rm = TRUE),
        R = sum(!is.na(beta_predicted)),
        Min = min(beta_predicted, na.rm = TRUE),
        Q1 = quantile(beta_predicted, 0.25, na.rm = TRUE),
        Median = median(beta_predicted, na.rm = TRUE),
        Q3 = quantile(beta_predicted, 0.75, na.rm = TRUE),
        Max = max(beta_predicted, na.rm = TRUE),
        
        .groups = 'drop'
      ) %>%
      mutate(
        SE = SD/ sqrt(R),
        Z2_stat = ifelse(SD != 0, (Mean - true_value_beta1) / (SE), NA),
        p_value_Z2 = ifelse(is.na(Z2_stat), NA, 2 * (1 - pnorm(abs(Z2_stat)))),
        CI_lower_Z = Mean - qnorm(0.9975) * (SE),
        CI_upper_Z = Mean + qnorm(0.9975) * (SE),

      ) %>%
      mutate(q = as.numeric(gsub("\\D", "", archivo))) %>%
      arrange(q) %>%
      select(q, everything(), -archivo)

    output_stats_path <- file.path(graph_folder, paste0("beta_stats_", subfolder, "_standard.csv"))
    write.csv(beta_stats, output_stats_path, row.names = FALSE)
    
    # Save beta stats for Capacity-constrained models
    beta_stats_cc <- combined_df %>%
      filter(model != "Standard") %>%
      group_by(archivo) %>%
      summarise(
        Mean = mean(beta_predicted, na.rm = TRUE),
        SD = sd(beta_predicted, na.rm = TRUE),
        R = sum(!is.na(beta_predicted)),
        Min = min(beta_predicted, na.rm = TRUE),
        Q1 = quantile(beta_predicted, 0.25, na.rm = TRUE),
        Median = median(beta_predicted, na.rm = TRUE),
        Q3 = quantile(beta_predicted, 0.75, na.rm = TRUE),
        Max = max(beta_predicted, na.rm = TRUE),
        .groups = 'drop'
      ) %>%
      mutate(
        SE = SD/ sqrt(R),
        Z2_stat = ifelse(SD != 0, (Mean - true_value_beta1) / (SE), NA),
        p_value_Z2 = ifelse(is.na(Z2_stat), NA, 2 * (1 - pnorm(abs(Z2_stat)))),
        CI_lower_Z = Mean - qnorm(0.9975) * (SE),
        CI_upper_Z = Mean + qnorm(0.9975) * (SE),
        
      ) %>%
      mutate(q = as.numeric(gsub("\\D", "", archivo))) %>%
      arrange(q) %>%
      select(q, everything(), -archivo)
    
    output_stats_path_cc <- file.path(graph_folder, paste0("beta_stats_", subfolder, "_cc.csv"))
    write.csv(beta_stats_cc, output_stats_path_cc, row.names = FALSE)
    
    # Boxplot
    combined_df <- combined_df %>%
      mutate(num_digits = nchar(str_extract(archivo, "\\d+"))) %>%
      arrange(desc(num_digits), desc(archivo))
    
    n_observaciones <- combined_df %>%
      group_by(valor_q, model) %>%
      summarise(n = n(), .groups = 'drop')
    
    # X-axis limits to improve boxplot presentation
    lower_limit <- -3 
    upper_limit <- 7
    
    palette_lines <- c("#ee5f2e","darkblue") 
    palette_fill <- muted(palette_lines, l = 80) 
    
    p <- ggplot(combined_df, aes(y = factor(valor_q, levels = unique(valor_q)), 
                                 x = beta_predicted, fill = model, colour = model)) +
      geom_boxplot(width = 0.5, outlier.shape = NA, size = 0.5) +
      scale_color_manual(values = palette_lines, name = "Model", 
                         breaks = c("Standard","Capacity-constrained"),
                         labels = c("Capacity-constrained" = "Capacity-\nconstrained", 
                                    "Standard" = "Standard\n ")) +
      scale_fill_manual(values = palette_fill,
                        breaks = c("Standard","Capacity-constrained"),
                        guide = "none") + 
      labs(title = "", y = "Capacity(q)", x =  expression(plain(beta))) +
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
      geom_vline(xintercept = true_value_beta1, color = "gray20", linetype = "dotted", size = 1) +
      scale_x_continuous(
        limits = c(lower_limit, upper_limit),
        breaks = seq(lower_limit, upper_limit, by = 1)
      ) +

      guides(color  = guide_legend(override.aes = list(size = 15))) + 
      guides(color = guide_legend(override.aes = list(fill = palette_fill, size = 0)))
    
    output_plot_path <- file.path(graph_folder, "boxplot_beta_predicted.pdf")
    ggsave(output_plot_path, plot = p, width = 30, height = 32, units = "cm")
  }
}

# Run function
graficas_box_plot("homog")
graficas_box_plot("heterog")
