################################################################################
# POLOE: Probability of observing at least one exhaustion
# Graph generation
################################################################################

rm(list=ls())
# Libraries
library(readr)
library(tidyverse)
library(ggplot2)

# Data graphing function
capacity_plot <- function(J_value, degree_differentiation_value) {
  
  # Data preparation
  graph_folder <- "C:/Projects/ProductAvaility/Repository"
  probs_n_nonexh <- read_csv(file.path(graph_folder, "probs_n_nonexh_full.csv"))
  
  df <- probs_n_nonexh %>%
    filter(J == J_value) %>% 
    filter(categoria == degree_differentiation_value) %>% 
    mutate(q = paste(q_capacity, sep = "")) %>%
    mutate(num_digits = nchar(str_extract(q, "\\d+"))) %>%  
    arrange(num_digits, q_capacity) %>%
    filter(!q %in% c(1, 100))
  
  # Plot
  g <- ggplot(df, aes(x = n_consumers,
                      y = (1 - cdf_value),
                      group = factor(q, levels = unique(q)),
                      color = factor(q, levels = unique(q)))
              ) +
    geom_line(size = 1) +
    scale_color_grey(start = 0.1, end = 0.8) +
    labs(
      title = "",
      x = expression(n),
      y = expression(P(J[n] != J ~ "|" ~ beta * "," ~ J * "," ~ q * "," ~ n)),
      color = "q =",
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),  
      strip.background = element_blank(), 
    )+
    theme(
      legend.key.width = unit(1, "cm"), 
      legend.key.height = unit(1, "cm"),  
      legend.text = element_text(size = 14), 
      legend.box.margin = margin(0, 0, 0, 0, "cm"),
      legend.margin = margin(0, 0, 0, 0, "cm"),
      legend.box.just = "center",
      legend.title = element_text(size = 16),
      legend.position = "bottom",
      legend.box = "vertical",
      axis.title.x = element_text(size = 14, face = "bold"),  
      axis.title.y = element_text(size = 14, face = "bold")
    ) +
    theme(panel.spacing = unit(.05, "lines"),
          panel.border = element_rect(color = "black", fill = NA, size = 1), 
          strip.background = element_rect(color = "black", size = 1)) +
    scale_x_continuous(
      limits = c(0, 101),  
      breaks = seq(0, 100, by = 10) 
    ) +
    geom_vline(xintercept = seq(0, max(df$n_consumers), by = 10), linetype = "dotted", color = "gray") +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray")  +
    guides(colour = guide_legend(ncol = 7, byrow = TRUE, reverse = F))
  
  output_folder <- file.path(paste(graph_folder,"graphs_capacity",sep="/"))
  
  if (!dir.exists(output_folder)) {
    dir.create(output_folder)
  }
  
  output_path <- file.path(paste(output_folder, sep="/"), paste0(unique(df$categoria), ", J=", unique(df$J),"-one.pdf"))
  ggsave(output_path, plot = g, width = 20, height = 12, units = "cm")
}

# Run function

capacity_plot(5,"homog")
capacity_plot(5,"heterog")
capacity_plot(100,"homog")
capacity_plot(100,"heterog")
