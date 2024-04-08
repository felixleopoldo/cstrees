library(stagedtrees)
#library(bnlearn)
library(parallel)
library(doParallel)


estimate_joint_distribution <- function(data) {

    cardinalities <- data[1, ]

    ### search the order
    data <- data[-1, ] ## remove first row
    
    model <- stagedtrees::search_best(data, alg = stages_bhc)

    # Generate all the outcomes in the order specified by the data labels
    spaces <- list()
    i <- 1
    for (c in cardinalities) {
        spaces[[colnames(data)[[i]]]] <- seq(c) - 1
        i <- i + 1
    }
    space <- rev(expand.grid(rev(spaces)))

    ## generate the full sample space
    prob <- prob(model, space)
    log_prob <- prob(model, space, log = TRUE)
    df <- cbind(space, prob = prob, log_prob = log_prob)
}

estimate_joint_distribution3 <- function(data) {

    cardinalities <- data[1, ]

    ### search the order
    data <- data[-1, ] ## remove first row


    # Estimate the model
    data <- data.frame(lapply(data, as.factor)) ## convert to factor
    dag <- bnlearn::tabu(data)
    dagfit <- bnlearn::bn.fit(dag, data)
    model <- as_sevt(dagfit) |> sevt_fit(data, lambda = 0) |> stages_bhc()

    # Generate all the outcomes in the order specified by the data labels
    spaces <- list()
    i <- 1
    for (c in cardinalities) {
        spaces[[colnames(data)[[i]]]] <- seq(c) - 1
        i <- i + 1
    }
    space <- rev(expand.grid(rev(spaces)))

    ## generate the full sample space
    prob <- prob(model, space)
    log_prob <- prob(model, space, log = TRUE)
    df <- cbind(space, prob = prob, log_prob = log_prob)
}

estimate_joint_distribution_pc <- function(data) {

    cardinalities <- data[1, ]

    ### search the order
    data <- data[-1, ] ## remove first row


    # Estimate the model
    data <- data.frame(lapply(data, as.factor)) ## convert to factor
  
    #get the dag using pc
    cpdag <- bnlearn::pc.stable(data)
    dag <- cextend(cpdag, strict=TRUE)

    dagfit <- bnlearn::bn.fit(dag, data)
    model <- as_sevt(dagfit) |> sevt_fit(data, lambda = 0) |> stages_bhc()

    # Generate all the outcomes in the order specified by the data labels
    spaces <- list()
    i <- 1
    for (c in cardinalities) {
        spaces[[colnames(data)[[i]]]] <- seq(c) - 1
        i <- i + 1
    }
    space <- rev(expand.grid(rev(spaces)))

    ## generate the full sample space
    prob <- prob(model, space)
    log_prob <- prob(model, space, log = TRUE)
    df <- cbind(space, prob = prob, log_prob = log_prob)
}

samp_size_range <- c(100, 1000)
seeds <- seq(10) - 1
num_levels_range <- c(5, 7, 10)
path <- "sim_results"


n.cores <- parallel::detectCores() - 1
#create the cluster
my.cluster <- parallel::makeCluster(
  n.cores, 
  type = "PSOCK"
  )

#check cluster definition (optional)
print(my.cluster)
#register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)
#check if it is registered (optional)
foreach::getDoParRegistered()
#how many workers are available? (optional)
foreach::getDoParWorkers()

x <- foreach(
  i = 1:10, 
  .combine = 'c'
) %dopar% {
    print("dsafafw")
    sqrt(i)
  }

  print(x)


# initiate empty data.frame with columns: alg, p, n_samples, seed, time
#df <- data.frame(method = character(), p = integer(), n_samples = integer(), seed = integer(), time = double())

for (num_levels in num_levels_range) {
    for (samp_size in samp_size_range) {
        #for (seed in seeds) {
        res <- foreach(seed = seeds, .combine = 'c') %dopar% {
            library(stagedtrees)
            library(bnlearn)
            library(parallel)
            library(doParallel)
            set.seed(seed)
            name <- paste0("p=", num_levels, "_n=", samp_size, "_seed=", seed, ".csv")
            data <- read.csv(paste0(path, "/data/", name), check.names = FALSE)

            st_est_path <- paste0(path, "/distr/stagedtrees/est/", name)
            print(st_est_path)
            # if file exists, skip
            if (file.exists(st_est_path)) {
                #next
            }

            dir.create(paste0(path, "/distr/stagedtrees/est"), showWarnings = FALSE, recursive = TRUE)
            dir.create(paste0(path, "/distr/stagedtrees/time"), showWarnings = FALSE, recursive = TRUE)
            
            start <- Sys.time()
            df <- estimate_joint_distribution(data)
            totaltime <- Sys.time() - start
            print(totaltime)            
            # write time to file
            tmp <- data.frame(method = "best order search", p = num_levels, n_samples = samp_size, seed = seed, time = totaltime)
            write.csv(tmp, file = paste0(path, "/distr/stagedtrees/time/", name), row.names = FALSE, quote = FALSE)
            
            write.csv(df, file = st_est_path, row.names = FALSE, quote = FALSE)
            tmp
        }
    }
    #write.csv(df, file = paste0(path, "/distr/stagedtrees/time.csv"), row.names = FALSE, quote = FALSE)
}

# Tabu first
for (num_levels in num_levels_range) {
    for (samp_size in samp_size_range) {
        for (seed in seeds) {
            set.seed(seed)
            name <- paste0("p=", num_levels, "_n=", samp_size, "_seed=", seed, ".csv")
            data <- read.csv(paste0(path, "/data/", name), check.names = FALSE)

            tabu_est_path <- paste0(path, "/distr/tabu/est/", name)
            print(tabu_est_path)
            # if file exists, skip
            if (file.exists(tabu_est_path)) {
                next
            }

            dir.create(paste0(path, "/distr/tabu/est"), showWarnings = FALSE, recursive = TRUE)
            dir.create(paste0(path, "/distr/tabu/time"), showWarnings = FALSE, recursive = TRUE)
            
            start <- Sys.time()
            df <- estimate_joint_distribution3(data)
            totaltime <- Sys.time() - start
            print(totaltime)            
            # write time to file
            tmp <- data.frame(method = "tabu", p = num_levels, n_samples = samp_size, seed = seed, time = totaltime)
            write.csv(tmp, file = paste0(path, "/distr/tabu/time/", name), row.names = FALSE, quote = FALSE)
            
            write.csv(df, file = tabu_est_path, row.names = FALSE, quote = FALSE)
        }
    }
    #write.csv(df, file = paste0(path, "/distr/stagedtrees/time.csv"), row.names = FALSE, quote = FALSE)
}

# PC first
for (num_levels in num_levels_range) {
    for (samp_size in samp_size_range) {
        for (seed in seeds) {
            set.seed(seed)
            name <- paste0("p=", num_levels, "_n=", samp_size, "_seed=", seed, ".csv")
            data <- read.csv(paste0(path, "/data/", name), check.names = FALSE)

            tabu_est_path <- paste0(path, "/distr/pc/est/", name)
            print(tabu_est_path)
            # if file exists, skip
            if (file.exists(tabu_est_path)) {
                next
            }

            dir.create(paste0(path, "/distr/pc/est"), showWarnings = FALSE, recursive = TRUE)
            dir.create(paste0(path, "/distr/pc/time"), showWarnings = FALSE, recursive = TRUE)
            
            start <- Sys.time()
            df <- estimate_joint_distribution_pc(data)
            totaltime <- Sys.time() - start
            print(totaltime)            
            # write time to file
            tmp <- data.frame(method = "pc + bhc", p = num_levels, n_samples = samp_size, seed = seed, time = totaltime)
            write.csv(tmp, file = paste0(path, "/distr/pc/time/", name), row.names = FALSE, quote = FALSE)
            
            write.csv(df, file = tabu_est_path, row.names = FALSE, quote = FALSE)
        }
    }
    #write.csv(df, file = paste0(path, "/distr/stagedtrees/time.csv"), row.names = FALSE, quote = FALSE)
}