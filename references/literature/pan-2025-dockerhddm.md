# Pan 2025 Adv Methods Pract Psychol Sci dockerHDDM a user-friendly environment for Bayesian hierarchical drift-diffusion modeling

> Text-layer conversion from `Pan_2025_Adv_Methods_Pract_Psychol_Sci_dockerHDDM_a_user-friendly_environment_for_Bayesian_hierarchical_drift-diffusion_modeling.pdf`. Verify tables and equations against
> the PDF before using exact values.

## Page 1

research-article20251298700 AMPXXX10.1177/25152459241298700Pan et al.Advances in Methods and Practices in Psychological Science





                                                                                                                                                               ASSOCIATION FOR
         Tutorial                                                                                    PSYCHOLOGICAL SCIENCE


                                                                                                                          Advances in Methods and
    dockerHDDM: A User-Friendly                                              PracticesJanuary-Marchin Psychological2025, Vol. 8,ScienceNo. 1,
                                                                                                                                    pp. 1­–26
                                                                                  © The Author(s) 2025    Environment for Bayesian Hierarchical                       Article reuse guidelines:
                                                                                                                                       sagepub.com/journals-permissions     Drift-Diffusion Modeling                                                                 https://doi.org/10.1177/25152459241298700DOI: 10.1177/25152459241298700
                                                                                                                               www.psychologicalscience.org/AMPPS


     Wanke Pan1   , Haiyang Geng2   , Lei Zhang3,4,5,6   , Alexander Fengler7,
      Michael J. Frank7, Ru-Yuan Zhang8   , and Hu Chuan-Peng1
           1School of Psychology, Nanjing Normal University, Nanjing, China; 2Tianqiao and Chrissy Chen Institute
            for Translational Research, Shanghai, China; 3Centre for Human Brain Health, School of Psychology,
            University of Birmingham, Birmingham, UK; 4Institute for Mental Health, School of Psychology, University
            of Birmingham, Birmingham, UK; 5Centre for Developmental Science, School of Psychology, University
            of Birmingham, Birmingham, UK; 6Social, Cognitive and Affective Neuroscience Unit, Department of
           Cognition, Emotion, and Methods in Psychology, Faculty of Psychology, University of Vienna, Vienna,
            Austria; 7Department of Cognitive and Psychological Sciences, Brown University, Providence, Rhode Island;
         and 8Brain Health Institute, National Center for Mental Disorders, Shanghai Mental Health Center, Shanghai
            Jiao Tong University School of Medicine and School of Psychology, Shanghai, China



        Abstract
          Drift-diffusion models (DDMs) are pivotal in understanding evidence-accumulation processes during decision-making
         across psychology, behavioral economics, neuroscience, and psychiatry. Hierarchical DDMs (HDDMs), a Python library
          for hierarchical Bayesian estimation of DDMs, has been widely used among researchers, including researchers with
         limited coding proficiency, in fitting DDMs and other sequential sampling models. However, issues of compatibility in
          installation and lack of support for more recent Bayesian-modeling functionalities pose serious challenges for new users,
          limiting broader adaptation and reproducibility of HDDMs. To address these issues, we created dockerHDDM, a user-
          friendly computational environment for HDDMs with new features. dockerHDDM brings three improvements: (a) easy
         to install once docker is installed, ensuring reproducibility and saving time for researchers; (b) compatible with machines
        with Apple chips; (c) seamless integration with ArviZ, a state-of-the-art Bayesian-modeling library. This tutorial serves
         as a practical, hands-on guide for researchers to leverage dockerHDDM’s capabilities in conducting efficient Bayesian
          hierarchical analysis of DDMs. The notebook presented here and in the docker image will enable researchers with
         various programming levels to model their data with HDDMs.


       Keywords
      HDDM, drift-diffusion models, Bayesian hierarchical modeling, reproducibility, docker, Python, open data,
       open materials

          Received 4/15/24; Revision accepted 10/21/24


       The drift-diffusion model (DDM) is one of the most    According to the DDM, experimentally observed pairs
        widely used computational models (for an overview, see    of response times and choices arise from a process of
           Ratcliff et al., 2016) to quantify the evidence-accumulation
        processes during decision-making  in neuroscience    Corresponding Authors:
                                                           Hu Chuan-Peng, School of Psychology, Nanjing Normal University,
       (Cavanagh et al., 2011; Herz et al., 2017; Shadlen &                                                                                      Nanjing, China
        Shohamy, 2016), psychology (Hu et al., 2020; D. J. Johnson     Email: hu.chuan-peng@nnu.edu.cn
          et al., 2017; Kutlikova et al., 2023), behavioral economics                                                                        Ru-Yuan Zhang, School of Psychology and Shanghai Mental Health
        (Desai & Krajbich, 2022; Sheng et al., 2020), and psy-     Center, Shanghai Jiao Tong University, Shanghai, China
         chiatry (Ging-Jehli et al., 2021; Pedersen et al., 2022).     Email: ruyuanzhang@sjtu.edu.cn


                              Creative Commons NonCommercial CC BY-NC: This article is distributed under the terms of the Creative Commons Attribution-NonCommercial 4.0
                             License (https://creativecommons.org/licenses/by-nc/4.0/), which permits noncommercial use, reproduction, and distribution of the work without
             further permission provided the original work is attributed as specified on the SAGE and Open Access pages (https://us.sagepub.com/en-us/nam/open-access-at-sage).

## Page 2

2                                                                                Pan et al.


                                             shown not only to correlate robustly with established
                                                          neural substrates (Chandrasekaran et al., 2017; Forstmann
               Upper Boundary                                             et al., 2016) but also to serve as a powerful measurement
                                                               tool for examining individual differences across cognitive
                 (v)                                   (a)              tasks, experimental manipulations, and participant popu-
            Rate                                               lations (Boag et al., 2024; Donkin & Brown, 2018; Evans        t            Drift                                      & Wagenmakers, 2020; but see Liu et al., 2023). Despite
   (z)                                                                                                                                                                               Threshold                  its theoretical contributions, the DDM is difficult to apply
    Bias                                                          to experimental data in practice because the derivation
               Lower Boundary                                         of inference-relevant quantities (e.g., the likelihood func-
                                    Time                         tion) requires a mathematical understanding of the com-
                                                        plex stochastic process of evidence accumulation.
Fig. 1.  Illustration of the evidence-accumulation process assumed by       Several software packages have been developed to
the drift-diffusion model (DDM). DDM has four basic parameters: drift     facilitate the application of DDM (see “Why Use docker­
rate (v), decision boundary (a), initial bias (z ), and nondecision time
(t ). The drift rate (v) is the average speed of evidence accumulation  HDDM Among Tools” section), proving particularly ben-
toward a decision; the decision boundary (a) is the distance between     eficial  for researchers with  limited computational
two decision thresholds, and the evidence needed to make a deci-    expertise. Among them, HDDM, a Python library for
sion increase as a increases; the initial bias (z ) reflects the starting    hierarchical DDM, is by far the most cited toolbox in the
point of evidence accumulation; nondecision time (t ) is the time not
used for evidence accumulation, for example, stimulus encoding or   community (Wiecki et al., 2013; with 996 citations in
motor execution. For a more detailed description of the DDM and its   Google Scholar as of August 26, 2024). Despite the suc-
parameters, see Table 1.                                       cess and popularity of HDDM,  it suffers from several
                                                               practical issues. First, the installation process of HDDM
stochastic evidence accumulation to a decision boundary     is cumbersome, exacerbated by its reliance on PyMC
(e.g., Voss et al., 2013; see Figure 1 and the related DDM    2.3.8 for Markov chain Monte Carlo (MCMC) sampling,
glossary in Table 1). This theoretical framework has been    a package that is no longer supported and may clash


  Table 1.  Drift-Diffusion Model Glossary

  Term                                                                 Description

   Accumulator            A component of the DDM that accumulates evidence for different decision options until a
                                  threshold is reached, triggering a decision.
  Random walk           A stochastic process that describes a path consisting of a sequence of random steps. It refers to
                                  the modeling of decision-making as a process of accumulating evidence over time.
   Diffusion                The diffusion refers to the variability in the evidence-accumulation process that represents
                           random fluctuations in the decision variable.
   Optional stopping         The concept of stopping the decision-making process at a point chosen by the decision maker,
                                  often when a certain level of confidence or evidence threshold is reached.

   Drift rate (v)              The average rate of evidence accumulation toward one of the decision boundaries. The more
                                          difficult the task, the less stimulus discrimination and the smaller the drift rate.
   Decision boundary (a)      The threshold that, when reached by the accumulated evidence, triggers a decision. It
                                  represents the speed-accuracy trade-off or caution, and the higher its value, the higher the
                                accuracy at the expense of slower response times.
   Nondecision time (t)       The time taken by processes other than decision-making (e.g., sensory encoding and motor
                                   execution). It simply shifts response time distribution.
    Initial bias (z)            The initial value of the decision variable, which indicates any initial bias in evidence
                                 accumulation, is also called ‘starting point’ in the literature. The closer it is to a boundary
                                   (1 and 0 correspond to the upper and lower boundaries, respectively), the faster and more
                                  frequent the response.
   Drift-rate variability (sv)     The variability in the drift-rate parameter across trials. It increases the proportion of slow errors.
    Initial bias variability       The across-trial variability in the initial bias parameter in the DDM. It increases the proportion of
       (sz)                               fast errors.
   Nondecision-time          The across-trial variability in the nondecision time parameter in the DDM. It simultaneously
       variability (st)                increases the probability of both faster and slower responses, resulting in a thicker tail of the
                          RT distribution.

   Note: The terms used here are defined within the framework of the sequential sampling model (Forstmann et al., 2016; Ratcliff et al., 2016),
   and some of them, such as diffusion and optional stopping, differ from those used in the mathematical literature. DDM = drift-diffusion
   model; RT = reaction/response time.

## Page 3

Advances in Methods and Practices in Psychological Science 8(1)                                              3


Table 2.  Comparisons Between dockerHDDM and the           (Fig. 2). To assist reproducibility and easy application, a
Original HDDM Package                                  corresponding step-by-step video walk-through is avail-
                                                         able on YouTube at https://www.youtube.com/watch?                     HDDM    dockerHDDM
                                                 v=ZU1fbXEuP8s or on OSF at https://osf.io/xz9m2.
Support ArviZa                 No           Yes            In the setup section (top panel in Fig. 2, correspond-
   Plotting (e.g., HDI)            No           Yes         ing to “Install Docker” section in this article), we provide
  Diagnosis (e.g., ESS)           No           Yes         instructions on how to install Docker. After that, we
  Model comparison (LOO-CV,     No           Yes        demonstrate how to obtain the dockerHDDM image and
     WAIC)                                         how to use this image to access the Jupyter notebook
Installation                      Hard          Easy
                                                              interface (middle panel in Fig. 2, corresponding to “Pull
Parallel processing                Hard          Easy
                                            dockerHDDM Image” and “Run dockerHDDM Container”
Compatibility with Apple chips      Hard          Easy
                                                               sections). Finally, within a working Jupyter notebook,
Note: HDI = high-density interval; ESS = effective sample size;       we show how to analyze an example data set with dock-
LOO-CV = leave-one-out cross-validation; WAIC, widely applicable     erHDDM in a principled Bayesian workflow (bottom
information criterion; PPC, posterior predictive checks.                                                       panel in Fig. 2, corresponding to “Example of Workflow”aPlotting, diagnosis, and model comparison are functions of ArviZ,
including HDI, ESS, LOO, WAIC, and PPC.                            section).

with the latest computer modules. Second, and for the   Install and use dockerHDDM
same reason, out-of-the-box HDDM is not compatible
with Apple chips, which creates a significant barrier for   Install Docker
Mac users. Third, although HDDM natively centers
                                                  Docker serves to create an all-in-one, fast, cross-platformaround Bayesian methods, it does not conveniently sup-
                                                  computing environment. The Docker website providesport all aspects of the evolved standards in Bayesian-
                                                         easy-to-follow installation instructions (https://docsmodeling workflows (Ahn et al., 2017; Gelman et al.,
                                                     .docker.com/get-docker/) and supports Windows,2020; Kruschke, 2021). Significant progress has recently
                                                MacOS, and Linux (see Box 2). Windows users shouldbeen made in supporting the principled Bayesian-mod-
                                                     ensure their system version is 21H2 (build 19044) oreling workflow in easy-to-use tool kits, such as the
                                                       higher and have either WSL or Hyper-V configuredPython package ArviZ (Kumar et al., 2019). Bridging
                                                       before installation (see https://docs.docker.com/deskthese new capabilities with HDDM facilitates a one-stop
                                                             top/install/windows-install/).Bayesian-modeling pipeline for experimentalists and
                                                                 After installing Docker Desktop (or Docker Enginecomputational modelers interested in applying the DDM
                                                                 for Linux users), one can verify the installation by run-to their experimental data.
                                                       ning the following command in a terminal (Fig. 3). If the  To address the above issues, we leveraged the Docker
                                                          container starts and runs successfully, it will display acontainer technology to create dockerHDDM, a stable
                                                        confirmation message and then exit (Fig. 3):and complete virtualized Python computing environ-
ment that enables out-of-the-box implementations of
Bayesian hierarchical DDMs. dockerHDDM has three   $ docker run hello-world
major advantages (Table 2). First,  it benefits from the
easy-to-deploy nature of the Docker environment to   Pull dockerHDDM image
avoid compatibility issues. Second, it is compatible with
both Intel and Apple chips. Third, it augments HDDM    After ensuring that Docker has been successfully installed
with ArviZ, a Python module that enables a wide range   and the Docker engine is running (Fig. 3), you can pull
of advanced Bayesian-modeling analyses. We expect    the dockerHDDM image by simply running the com-
dockerHDDM to provide an easy-to-use environment to   mand in the terminal (see the meaning of each argument
help researchers across various backgrounds efficiently    in Fig. 4a):
use DDM in their research.
                                                 $ docker pull hcp4715/hddm
How to Follow This Tutorial                                                          or
The primary goal of this article is to present a practical
guide to dockerHDDM for beginners with little modeling   $ docker pull hcp4715/hddm:latest
experience. In the tutorial, we start with step-by-step
instructions on how to configure the dockerHDDM envi-      This command will pull the latest default version of
ronment and how to use  it in practical data analysis   dockerHDDM, which corresponds to the image with the

## Page 4

4                                                                                Pan et al.

     1. Install docker                                1.1 To download and install docker, one should follow the official
                            Docker instructions exactly (https://docs.docker.com/get-docker/).
                                1.2 To test the docker installation by command line:
                               docker run hello-world
                              Note: the Windows user should configure WSL in advance (Windows
                            Subsystem for Linux)(https://docs.docker.com/desktop/wsl/).

     2. Pull and run dockerHDDM
       2.1 Open terminal & Run command line           2.2 Open URL & Enter Jupyter
                                 Terminal
       user@DESKTOP:/$ docker pull hcp4715/hddm
                                                                                                     http://127.0.0.1:8888/?token=0ce749eb...       user@DESKTOP:/$ docker run -it --rm -p 8888:88
      88 -v $(pwd):/home/jovyan/work hcp4715/hddm j
       upyter notebook

       [C 06:50:52.342 NotebookApp]
           To access the notebook, open this file in a
       browser:
           ...
           Copy and paste URL:
             http://127.0.0.1:8888/?token=0ce749eb...

     3. HDDM analysis workflow





  Fig. 2.  dockerHDDM usage flowchart. The code in the figure is for demonstration purposes only. Specific instructions and copyable code
  can be found in the following corresponding sections. The top panel describes how to install Docker, corresponding to “Install Docker”;
  the middle panel describes how to pull and run dockerHDDM, corresponding to “Pull dockerHDDM Image” and “Run dockerHDDM Con-
   tainer”; and the bottom panel shows the workflow in dockerHDDM, corresponding to “Example of Workflow.” A video tutorial is available
   at https://www.youtube.com/watch?v=ZU1fbXEuP8s and https://osf.io/xz9m2.

## Page 5

Advances in Methods and Practices in Psychological Science 8(1)                                              5


Box 1.  Glossary of Terms Used in Bayesian Modeling

 Prior, or prior distribution, often referred to as p( θ ) , is the initial belief that researchers have from pilot data.
    Likelihood, or likelihood function, often referred to as p ( y | θ ) , is the probability of the observed data y
 as a function of the specific parameters θ of a chosen statistical model. For example, the Bernoulli function is
 the likelihood function for statistically describing coin tossing.
    Posterior, or posterior distribution, often referred to as p ( θ | y ) , refers to the updated beliefs about the
 parameters θ after observing the data y, balancing prior knowledge with observed data according to Bayes’s
  rule, that is, p ( θ | y ) ∝ p ( y | θ ) p ( θ ) .
   Markov chain Monte Carlo (MCMC) is a sampling method to infer the posterior distribution by
 simulation. The Markov chains (usually multiple MCMC chains are required) are algorithmically constructed so
  that their corresponding stationary distribution using MCMC samples matches the posterior distribution of
  interest in the limit (Kruschke, 2014; Robert & Casella, 2004). The process of reaching this stationary
 distribution is called “MCMC convergence.” These sampled parameter values serve as the approximation to the
 posterior distribution and can then be used to obtain empirical estimates of the posterior distribution and
 associated summary statistics of interest using Monte Carlo integration. In the literature, a chain (or trace) is
 referred to as a collection of samples (or draws). Traces serve as a basis for diagnosing convergence and/or
 other potential problems with the procedure in a given application. MCMC is particularly useful for models
 with high complexity.
    Effective sample size (ESS) is the number of independent samples with the same estimation power as
 the N autocorrelated samples from each MCMC chain. ESS is often used to determine whether the number of
 draws in MCMC chains is sufficient to guarantee a reliable estimate of uncertainty. An ESS greater than 400
  is recommended, with the ESS for all four MCMC chains being 100 (Vehtari et al., 2021). However, the
 required ESS should be informed by the statistics one wishes to estimate from the posterior. It is
 recommended that an ESS of at least 10,000 is required for reasonably stable estimates of highest density
  intervals; for stable estimates of equal-tailed intervals, a lower ESS is sufficient; a smaller ESS may yield
 stable estimates of the central tendency, especially if it falls in a high-density region of the distribution
 (Kruschke, 2018, 2021).
   Gelman-Rubin statistics ( ˆR) is the ratio of within-chains variability to between-chains variability. Values
 close to 1.0 for all parameters and quantities of interest suggest that the MCMC algorithm has sufficiently
 converged to stationary distributions. In practice, the maximum ˆR must be less than 1.1 (Annis et al., 2017),
 more stringent criteria requires the ˆR values of less than 1.01, and a compromise is 1.05
 (A. A. Johnson et al., 2022).
    Posterior predictive samples, p ( y| y ), simulates new data y conditional on the posterior distribution
 given the observed data y. The simulated data can then be used to check whether the model can be
 considered a good fit to the data-generating mechanism by comparing the simulation with the observed data.
 This process is often called “posterior predictive checks.”
   Leave-one-out cross-validation is a model-evaluation approach in which the model is trained on all
 observations except for a single observation yi (where i = 1, 2, 3, … , n ), and then used to predict the held-out
 observation yi. This procedure is repeated for each of the n observations.
   Log predictive density, log p ( y|θ), is an overall summary of a model’s predictive abilities by estimating
 the log-likelihood of new data  y given the true parameters θ. However, because both the new data  y and the
 true-model parameters θ are typically unavailable in empirical data, the log predictive density is approximated
 using the observed data y and the posterior estimates of the parameters ˆθ, hence log p ( y| θ ) ≈ log p ( y| θˆ ) . This
 estimate, when multiplied by –2, gives the deviance, −2 log p ( y|θˆ ). However, because log p ( y|θˆ ) is a biased
 estimate of log p ( yθ|  ), an adjustment is required to correct the bias.
                                                      n      1      S                   s     Log pointwise predictive density,                                                   log               ∑ i =1                   ∑ s =1                                                                                                                      S      p ( y i |θ )  , is the likelihood of each observed data
 point yi conditional on the model parameters θs. In practice, this quantity is estimated using samples θs (for
  s = 1, 2, 3, … , S ) drawn from the posterior distribution.


                                                                                               (continued)

## Page 6

6                                                                                Pan et al.


Box 1.  (continued)


                                                                               n
   Expected log pointwise predictive density (ELPPD),                                                        E f ( log p post ( y i ) ), is a measure of out-of-                      ∑ i = 1
 sample predictive performance for new data  yi generated by the true data-generating process. p post ( y i ) is the
 predictive density for  yi based on the posterior distribution, f is the true underlying model, and E f denotes the
 expectation that averages over the true data-generating distribution (Gelman et al., 2014). ELPPD is commonly
 the unknown parameters θ in a model before observing data. It can either be formed from existing research or
 used to compare the predictive performance of different models because it provides an estimate of how well a
 model is expected to perform on new data.
   Highest density interval (HDI) is an estimate of a parameter’s credible range in the context of Bayesian
  statistics. It encompasses an interval of the posterior distribution in which each point within this interval has a
 higher density than points outside of it. For instance, a 95% HDI means that there is a 95% chance that the true
 parameter value falls within this range, making it a reliable indicator of parameter uncertainty. HDIs are commonly
 used for hypothesis testing regarding effect sizes and comparisons across different conditions or groups.
   A region of practical equivalence (ROPE) is a predefined range of parameter values that are considered
  practically equivalent to zero, which could be based on existing literature or theoretical reasoning (Kruschke,
 2018, 2021). To determine whether a parameter estimate is significantly different from zero, a ROPE might be
  set as a range around zero. If the 95% HDI of the parameter lies entirely outside this ROPE, the parameter is
 considered credibly different from zero. If the HDI is entirely within the ROPE, the parameter is effectively zero
  for practical purposes. Partial overlap suggests that the parameter’s result should be interpreted with caution.
 Note that caution should be taken when using the HDI + ROPE method for statistical inference on transformed
 parameters because of an inconsistency in transformation properties between ROPE and HDI (Etz et al., 2024).
   Bayes factor (BF) and Savage-Dickey density ratio (SDDR): BF quantifies the strength of evidence for
 one statistical model over another. A value greater than 1 suggests more support for the alternative model
  relative to the original model, offering a continuous measure of evidence (Kass & Raftery, 1995). The SDDR
 simplifies BF computation for nested models by comparing a parameter’s posterior density at a specific point
  (typically zero) to its prior density at the same point. This method is efficient and effective for evaluating
 whether a parameter is significantly different from zero (Wagenmakers et al., 2010).


tag `1.0.1`. One can also select different tags for     this article works with the `latest` or `1.0.1` tags,
different versions of HDDM (see https://hub.docker   and  it is compatible with 0.8.0, with minor grammar
.com/r/hcp4715/hddm/tags). Note that the tutorial in    changes.


Box 2.  Basic Introduction to Docker

 Docker is an open-source platform that automates the deployment, scaling, and management of applications.
  It achieves this through containerization, a process that packages an application and its dependencies into a
  single, portable, and consistent unit, known as a “container image.” Containers ensure that applications run
  reliably regardless of the environment (Peikert & Brandmaier, 2021; Wiebels & Moreau, 2021).
   Docker uses a client-server architecture in which the Docker client communicates with the Docker daemon,
 responsible for building, running, and distributing containers. The core components of Docker are the Docker
 Engine, Docker Hub, and Docker Compose. The Docker Engine is the runtime that enables containerization,
 and Docker Hub is a cloud-based registry for sharing and managing container images. Docker Compose, on
 the other hand, is a tool for defining and running multicontainer Docker applications.

 Common Docker Commands:
   `docker pull [image]`: Downloads a Docker image from a registry. For instructions on downloading
     the dockerHDDM image, see “Pull dockerHDDM Image.”
   `docker run [image]`: Runs a container from a Docker image. For details on how to start a container
     using the dockerHDDM image, see “Run dockerHDDM Container.”

                                                                                               (continued)

## Page 7

Advances in Methods and Practices in Psychological Science 8(1)                                              7


Box 2.  (continued)

   `docker images`: Lists all Docker images on the local machine. This can be used to check different
      versions of the dockerHDDM image.
   `docker commit [container_id] [new_image_name]`: Creates a new image from a container’s
     changes. For example, if you modify or install new Python packages in the dockerHDDM container, you
     can save these changes as a new image.
   `docker build [dockerfile]`: Builds a Docker image from a Dockerfile in the current directory.
    You can customize the dockerHDDM image using the provided Dockerfile.
   `docker push [repository/image:tag]`: Uploads a Docker image to a registry. After logging in,
     you can push the saved image to Docker Hub or any other Docker registry.
   `docker rmi [image]`: Removes a Docker image from the local machine. This is useful for cleaning
    up unused images.
   `docker save -o [output_file] [image]`: Saves a Docker image to a tar archive file. This is
      useful for backing up images or transferring them to another system.
   `docker load -i [input_file]`: Loads a Docker image from a tar archive file. This can be used to
      restore or import images from a backup.


Run dockerHDDM container                      platform. The `-v` option is used to mount a local
                                                           folder into the container’s  file system, enabling  file
After pulling the Docker image to a local machine, you can                                                  exchange from the host machine. The example code
start a computing environment by running the docker-                                             `$(pwd):/home/jovyan/work` specifies two paths
HDDM image with the command in the terminal (Fig. 4b):                                                        separated by a colon. The path on the left, denoted by
                                                `$(pwd)`, represents the current working directory on$ docker run -v $(pwd):/home/jovyan/work
                                                          the host machine, and the path on the right, `/home/-p 8888:8888 -it --rm hcp4715/hddm jupyter
                                               jovyan/work`,1 is the location inside the containernotebook
                                                where the folder will be mounted (Fig. 4b). This means
This command creates a Docker container, which is a    that you can read and write the files from your local
specialized environment encapsulated within the Docker   machine  in the “work” directory  in the browser.





Fig. 3.  Command to check Docker installation in terminal. After running the command `docker run hello-world` (highlighted at first line),
the printout shows that Docker has been successfully installed on the system. The schematic interfaces of the terminal on different platforms
are shown: (left) MacOS, (middle) Windows, and (right) Ubuntu.

## Page 8

8                                                                                Pan et al.

          a
                                                                         Using docker to execute this command

                                                                           Pull/download an image from docker hub
                                                                     Docker hub account that maintain the image

                                                                        Image’s name

                                                                       Handle of a tag of the image

                          docker   pull  hcp4715 /hddm  :latest
          b                                                                             Run a container
                                                                                Mount a volume,
                                                                                                  localPath:containerPath
                                                                     Map container port,
                                                                                                  hostPost:containerPort
                                                                                           Continue the command
                                                                                                                in a new line*

                        docker  run  -v    $(pwd):/home/jovyan/work   -p  8888:8888    \
                                                          -it --rm   hcp4715/hddm:latest     jupyter notebook

                                                                             Open jupyter notebook
                                                                                 The docker image (and its tag)
                                                                                                          to run the container
                                                                             Run container interactively
                                                                                          Clean up containers and
                                                                                                   delete ﬁles on container exit

                     Fig. 4.  Docker commands to download and run dockerHDDM. (a) Download/pull dockerHDDM
                   from the Docker hub. The command by default downloads the latest version of `hcp4715/dock
                 erHDDM` if the image tag is not specified. The CPU architecture (Apple or Intel chips, correspond-
                      ing to ARM64 and AMD64 architectures, respectively) is automatically recognized when the image
                             is downloaded. (b) Command to start a container. Note, “\” separates different lines of a command
                        in Linux and MacOS terminals but not in Windows.


`$(pwd)` can be replaced with a valid folder path on   `-p 7777:8888`); in this case, you should replace
your local machine. For example, for a folder named    the “8888” in the URL to “7777” (e.g., “http://127.0.0.1:7
“ddm_project” on the drive D, it can be mounted with    777/?token=. . .”). You can then open or initialize a Jupy-
the following arguments in the respective operating sys-     ter notebook2 to code, run, and view the output directly.
tems: in Linux, `-v /mnt/d/ddm_project:/home/   Note that the `--rm` flag included in the command
jovyan/work`; in Windows, `-v D:\ddm_project:/   means that the dockerHDDM container, along with any
home/jovyan/work`; and in MacOS, `-v /Volumes/D/    data or newly installed Python modules, will be deleted
ddm_project:/home/jovyan/work`. The other argu-   when the container stops. However, any files or data
ments in the command are explained in Fig. 4b.         mounted to the container from the `$(pwd)` path will
   After running the `docker run . . .` command,   remain unaffected. This ensures the reproducibility of
a URL appears at the end of the terminal output (Fig. 2,    the computing environment. If you wish to modify the
middle panel). You can copy and paste this URL “http://   computing environment, for example, by installing addi-
127.0.0.1:8888/?token=. . .” into any web browser (e.g.,    tional Python modules, we recommend that you first
Firefox or Chrome) to launch a Jupyter interface based    read the Docker API before removing `--rm` directly.
on the dockerHDDM container. If the URL does not load       In the Jupyter interface, you will find two files and
properly, check whether port 8888 is being used by   two folders (Fig. 2, middle). The notebook docker-
other Docker containers or programs.  If so, close   HDDM_Workflow.ipynb offers a detailed reproduction
those containers or programs. Alternatively, you may    of the analyses presented in this article, which we dis-
change the port, for example, use port 7777 (i.e., set    cuss further in “Example of Workflow.” In contrast, the

## Page 9

Advances in Methods and Practices in Psychological Science 8(1)                                              9


notebook dockerHDDM_Quick_View.ipynb provides a    provided two main parameters to set the MCMC algo-
brief overview of the dockerHDDM image’s new features    rithm; the first parameter was the number of samples
and an introduction to basic modeling processes. One    (`500`), and the second was the number of burn-ins
folder is “work,” which mounts the local path into the   (`burn=100`).3
docker environment. The other folder, “OfficialTutorials,”       In dockerHDDM, we included five extra arguments
contains notebooks that reproduce the official tutorials    in `.sample()` method to provide parallel computing
available at https://hddm.readthedocs.io/en/latest/tuto    for MCMC chains and create InferenceData.
rials.html. Beginners can follow HDDM_Basic_Tutorial.      To preserve compatibility and consistent output with
ipynb to get a basic understanding of HDDM, as dis-    origin HDDM, the arguments are configured with the fol-
cussed in Wiecki et al. (2013); HDDM_Regression_Stim    lowing defaults: `return_infdata=False`, `sample_
coding.ipynb covers more advanced models with regres-   prior=False`, `loglike=False`, `ppc=False`,
sion, in which parameters can vary based on experimental   `save_name=None`, and `chains=1`.
conditions and other covariates; Posterior_Predictive_     The `chains` argument determines the number of
Checks.ipynb introduces posterior predictive checks  MCMC chains. Using more than two chains triggers mul-
(PPCs), showing how to generate predicted data from    tithreaded parallel computation, which can significantly
fitted parameter posteriors and how to analyze these pre-    reduce the time when multiple chains are needed to
dicted data; LAN_Tutorial.ipynb introduces advanced use   compute model diagnosis index ˆR (see “Model Diagno-
of LAN functions that address the problematic likelihood     sis”). The number of parallel MCMC chains is limited by
of more complicated models based on neural-network    the number of available CPU cores/threads available.
methods (Fengler et al., 2021).                           For example, the maximum number of chains for a com-
                                                         puter with four cores (eight threads) is eight. Setting the
                                                           “chains” argument more than eight may degrade perfor-Novel Features of dockerHDDM                                                  mance. Nonetheless, whenever possible, a number of
The dockerHDDM_Quick_View.ipynb  illustrates two    four chains is commonly used.
novel features in dockerHDDM (compared with HDDM     The `return_infdata`argument converts HDDM
installed directly without Docker): parallel computing    results into the InferenceData structure,4 accessible via
for MCMC chains and creating InferenceData data for   `model.infdata`, by default set to `False` to main-
ArivZ analyses (as shown in the <Code Block 1>):          tain compatibility with original HDDM output. In addi-
                                                                  tion, we have included `loglike` for computing and
<Code Block 1>                                          saving log-likelihood values (see “Model Comparison”),
```Python                                      `ppc` for PPCs (see “PPC”), and `sample_prior=True`
# define a simple model with preloaded           for calculating Savage-Dickey density ratio (Wagenmak-
data                                                         ers et al., 2010) to approximate Bayes factor (BF; see
model = hddm.HDDM(data)                                    “Statistical Inference”). When setting `ppc` as `True`,
                                                                                           it defaults to generating 500 predictions  for each
# origin model fitting code                     observed data, but users can adjust this by adding argu-
# model.sample(500, burn = 100)               ment `n_ppc`. Likewise, when setting `sample_
                                              prior` as `True`, it defaults to sampling 2,000 draws
# dockerHDDM new model fitting code               for each prior parameter, but users can adjust this by
model.sample(                                      adding argument `n_prior`.
  500, burn = 100,                                             Finally, the `save_name` argument specifies the path
  chains = 4,  # parallel computing for       and file name for saving the model and InferenceData,
    MCMC chains                                  which is convenient for reusing results. One can load
  return_infdata = True,  # return               the model using `model = hddm.load(‘example.
    InferenceData for ArivZ analysis          hddm’)` and the InferenceData with `infdata  =
  sample_prior = True, loglike = True, ppc    az.from_netcdf(‘example.nc’)`.
    = True,
  save_name = ‘example’                                      Example of Workflow)
```                                                       In this section (Fig. 2, bottom panel), we demonstrate
                                         how to use dockerHDDM (i.e., HDDM and ArviZ) to
  For all models defined by methods such as `hddm.    perform key steps of Bayesian modeling (Gelman et al.,
HDDM()` or `hddm.HDDMRegressor()`, the user    2020; Martin et al., 2024): model specification and fitting,
can employ the `.sample()` method to run the   model diagnosis, model comparison, PPC, and statistical
MCMC algorithm for model fitting. The original HDDM    inference. The code reproduced in this section can be

## Page 10

10                                                                               Pan et al.


Table 3.  Example Data Set From Cavanagh et al. (2011)        options define two levels of conflict: high conflict for
                                                     low-low and high-high trials (“HC” in variable “conf”) and
subj_idx                rt             response            conf
                                                 low conflict for low-high trials (“LC” in variable “conf”).
0                 1.21                 1.0           HC        Note that HDDM requires the inclusion of three col-
0                 1.63                 1.0             LC     umns of variables, “subj_idx,” “rt,” and “response,” to con-
0                 1.03                 1.0           HC       struct the hierarchical model. This means that when
0                 2.77                 1.0             LC      analyzing your own data, these three columns of variables
0                 1.14                 0.0           HC     must appear in the data set with identical column names.
                                                             In addition, the unit of “rt” must be seconds, and “response”Note: The data structure required for HDDM is long-format data, where
each row represents one trial. “subj_idx” is the subject index, “rt” is          is coded as 1 for the upper boundary of the corresponding
the response time (in seconds), and “response” in this case represents     choice and 0 for the lower boundary (for more details, see
the accuracy, where 1 is correct and 0 is incorrect. These three           https://hddm.readthedocs.io/en/latest/howto.html).
columns of data are mandatory when using HDDM and must be kept
consistent with the column names and the units (rt, seconds). “conf” is
an optional variable, corresponding to the conflict level, where “HC”    Model Specificationdenotes high conflict and “LC” denotes low conflict. “conf” is not a
mandatory variable or column, meaning that different factor names and    As a demonstration of model specification, we test an
levels can be used depending on the experimental design. In addition,
multiple variables may be maintained in the data, which may be        example question: Is there an effect of conflict levels on
categorical or continuous.                                                  drift rate (Wiecki et al., 2013). To answer the question, we
                                                          constructed three computational models (see Table 4).
found in dockerHDDM_Workflow.ipynb in the docker-     Model 0 served as the baseline without considering
HDDM environment.                                      the effect of conflict level on the model parameters. The
                                                model contains the seven parameters, referred to as the
                                                                          full DDM, including the decision boundary (a), drift rate
Example Data                                                                     (v), nondecision time (t), decision bias (z), and sv, st,
For convenience, we use the data from Cavanagh et al.   and sz , which indicate the trial-by-trial variations of v,
(2011), which is built within HDDM, as an example to      t, and z ( Boehm et al., 2018; Ratcliff & Tuerlinckx, 2002 ) .
demonstrate how to implement the modeling workflow.     By default, HDDM considers the hierarchical-model-
This data set contains response time and choice data from    ing approach that includes parameters at both the indi-
14 Parkinson’s patients (see Table 3). In the experiment,    vidual and the group levels (see Box 3). Model 0 has 11
participants were asked to choose between two options    population-level parameters, including the means and
associated with either high or low reward values (i.e.,    the standard deviations for the four basic parameters
reward probabilities in typical reinforcement-learning    (a/v/t/z) and three parameters (sv/st/sz ) for the inter-
tasks). The relative value differences between the two     trial variations. At the individual level, each subject also


          Table 4.  Models Used in This Tutorial

                                  HDDM functions for defining a model (`df`
           Models              Describe                       is the data from Cavanagh et al., 2011)      n params

           Model 0     Baseline                   hddm.HDDM(df, include=[‘a’, ‘v’,            67
                                                ‘t’, ‘z’, ‘sv’, ‘sz’, ‘st’])
           Model 1     Varying drift rates across      hddm.HDDM(df, include=[‘a’, ‘v’,            82
                            conditions                 ‘t’, ’z’, ‘sv’, ‘st’, ‘sz’],
                                                depends_on={‘v’: ‘conf’})
           Model 2     Varying within-subjects drift    hddm.HDDMRegressor(df, “v ~ 1               83
                               rates across conditions        + C(conf, Treatment(‘LC’))”,
                                                group_only_regressors=False,
                                                keep_regressor_trace=True,
                                                include=[‘a’, ‘v’, ‘t’, ‘z’, ‘sv’,
                                                ‘st’, ‘sz’])

              Note: `hddm.HDDM()` is the default function for constructing a hierarchical drift-diffusion model. The `include`
             argument allows the addition of free parameters, which are fixed by default. The `depends_on` argument specifies a
             parameter (e.g., v) that depends on a categorical independent variable (e.g., ‘conf’). The `hddm.HDDMRegressor()`
                   is an HDDM function that includes effects of conditions in a linear regression fashion. The `keep_regressor_trace`
            argument allows a trace of the regressor to be kept, which is needed for posterior predictive checks. By default, the
               hierarchical regression allows only the intercept to vary across participants, and the slope is fixed at the population
                 level. The `group_only_regressors = FALSE` argument additionally estimates the slopes at the individual level in
              the regression model.

## Page 11

Advances in Methods and Practices in Psychological Science 8(1)                                             11

has a full set of four basic parameters, yielding a total   we use a hierarchical regression model with `hddm.
of 56 = 14 × 4 parameters. Thus, Model 0 has 11 + 56 = 67   HDDMRegressor()` by using the formula `v ~ 1 +
free parameters.                                 C(conf, Treatment(‘LC’))` (see Box 3). This for-
  Model 1 allows the drift rate to vary as a function of    mulation automatically assigns two free parameters, the
the conflict levels (i.e., `depends_on={‘v’: ‘conf’}`    intercept and slope, to each subject. Thus, there are
in HDDM). Specifically, Model 1 sets two drift-rate vari-    5 × 14 = 70  individual-level parameters  in Model  2.
ables each for low- and high-conflict levels at both the    Accordingly, Model 2 has four parameters for v: “v_Inter-
population and individual levels, respectively. Thus,    cept” and “v_Intercept_std” are the mean and standard
Model 1 has 12 population-level parameters: the means    deviation of the intercept, and “v_C(conf)[T.HC]” and
and standard deviations for a,  t , and z; two means    “v_C(conf)[T.HC]_std” are the mean and standard devia-
(“v_(LC)” and “v_(HC)”) and one standard deviation for    tion of the slope. Therefore, Model 2 has 13 population-
v; and three intertrial variability parameters (sv/st/sz ).    level parameters: the means and standard deviations for
Likewise, at the individual level, there are 5 (vLC/vHC/t/z/    a, t, and z; the means and standard deviations of the
a) × 14 (subjects) = 70 individual-level parameters. Thus,    slope and the intercept of the regression for v; and three
Model 1 has a total of 82 free parameters.                     intertrial variability parameters (sv/st/sz). Taken together,
  Note that Model 1 assumes complete independence   Model 2 has a total of 13 + 70 = 83 free parameters.
between high and low levels of conflict within subjects.
This assumption may be inappropriate because  it  is                                      Model fittinglikely that a person who responded relatively quickly in
the “LC” condition will also respond relatively quickly   The defined HDDM model allows the MCMC algorithm
in the “HC” condition and vice versa. For more detailed    to be run using the `.sample()` method for model
differences between Model 1 and Model 2, see Box 3.      fitting and parameter estimation. The definition and fit-
  Model 2 was constructed to include correlations    ting of Model 2 are used here as an example (see <Code
between drift rates across conflicting levels. In Model 2,    Block 2>):


Box 3.  Parameters in Hierarchical Drift-Diffusion Models

 HDDM employs hierarchical Bayesian modeling by default, where each participant’s free parameters are sampled
 from population-level distributions (Wiecki et al., 2013). Taking full drift-diffusion model (DDM; Model 0) as an
 example, nondecision time tp is assumed to be drawn from a normal distribution: t p ~ N ( µt ,σt ), where µt and σt
  are the mean and standard deviation of the population-level normal distribution of nondecision time t. Likewise,
 uz / uv / ua and σz /σv /σa are the means and standard deviations for the other three parameters, respectively. In
  addition, three free parameters st /sv / sa indicate the trial-by-trial variability of nondecision time (t), drift rate (v),
 and initial bias (a), which are estimated only at the population level.

                           Full DDM: hddm.HDDM(data, include=[‘z’, ‘sv’, ‘sz’, ‘st’])

                        µν      σν         µa      σa          µz      σz           µt       σt


                                   νp                   ap                        zp                               tp          sν


                                                                                                                                       st
                                                                                             Xi,p
                                                                                                                 i = 1, …, Np                                            sz
                      p = 1, …, P


                          Note: The hierarchical structure of the full DDM in HDDM. The parameters inside and
                           outside the rectangle are subject and population level parameters, respectively. p i/ are
                           the indices of participants (p = 1, 2,..., P ) and trials (i = 1, 2, …. N ), where xi , p is the data
                          (choice/response time) of the ith trial in the pth subject.

                                                                                               (continued)

## Page 12

12                                                                               Pan et al.


Box 3.  (continued)

    Consequently, there are a total of 11 population-level parameters. At the subject level, subjects have their
 own estimate of the parameter of a, v, t, z, leading to a total of 4 × p subject-level parameters. Thus, in the full
 DDM, the number of parameters is 11 plus 4 × p.
  HDDM provides two types of priors: weakly informative priors and noninformative priors. By default,
 dockerHDDM uses weakly informative priors as summarized in the table below (Wiecki et al., 2013). The default
 informative priors are suitable for most perceptual tasks. However, for tasks with longer response times, it is
 recommended to use noninformative priors. In this case, one has to set the parameter `informative=False`
 when defining the model, for example, `m = hddm.HDDM(data, informative=False)`.

                                           

                  DDM parameters’ informative prior
                             mv~ 2(  , 3 )                     σv ~ HN ( 2 )       v p ~ µ(  v , σ2v )
                         ma~ 1(  . 5, 0. 75 )               σa ~ HN ( 2 )      ap ~ µ(  a , σ2a )
                              mz ~ invlogit ( 0(  . 5, 0. 5 ))      σz ~ HN ( 0. 05 )    z p ~ ( µ z , σ2z )
                            mt~ 0(  . 4, 0. 2 )                     σt ~ HN (1)            t p ~ µ(   t , σ2t )
                                 sv ~ HN ( 2 )                            st ~ HN ( 0. 3 )        sz ~ 1( , 3 )

                                 Note: Table extracted and refined from Wiecki et al. (2013).  represents a
                              normal distribution parameterized by the mean (m) and standard deviation (σ).
                  HN represents a half-normal distribution, which is a positive-only distribution
                                 parameterized by the standard deviation.  represents a gamma distribution,
                                parameterized by the mean (m) and the rate (σ).  represents a beta
                                       distribution, parameterized by alpha and beta. The term invlogit represents
                                  the inverse logit function also known as the logistic function.

  HDDM also allows parameters to vary with variables by integrating hierarchical linear regression models
 (also called “linear mixed models” or “multilevel models”). Specifically, the `hddm.HDDMRegressor()`
 function allows any or all of the four parameters of DDM (a, v, t, z) to be modeled as a function of
 experimental conditions or other variables (e.g., EEG signal). In HDDM, the regression models are defined
 using the Python package patsy (see https://patsy.readthedocs.io/en/latest/quickstart.html), which uses the
 same syntax for defining regression functions as in other commonly used statistical packages. For example, in
 Model 2 in the main text, we used the expression `v ~ 1 + C(conf, Treatment(‘LC’))`, where the term
 to the left of “~” is the dependent variable and the term to the right of “~” is the regression equation. The term
  ‘1’ refers to the intercept, which corresponds to the variable “V_Intercept” in the output. The term “C(conf,
 Treatment(‘LC’))” indicates the slope coefficient, which corresponds to the variable “v_C(conf, Treatment(‘LC’))
 [T.HC]”. As in other hierarchical regression models, both the intercept and the slope can be estimated at the
 population level and the subject level (referred to as “fixed effects” and “random effects” or “varying effects,”
 respectively; D. J. Johnson et al., 2017; Pedersen & Frank, 2020; Wiecki et al., 2013), depending on how the
 model is specified. In `hddm.HDDMRegressor()`, the default is hierarchical model with random intercept
 but no random slope. We need to set `group_only_regressors=False` to include the random slope (as
 we did in Model 2).
    Although both the `depends_on` argument and the `HDDMRegressor` function allow parameters to vary
 with discrete variables (e.g., conflict levels), there is an important difference between them. The `depends_
 on` argument defines the parameter split by condition. Specifically, the means of the parameters under each
 condition are derived from a share prior, whereas the variability of the parameters is consistent across
 conditions. The `HDDMRegressor` function defines the relation between parameters and condition by a
 linear model specification, which means the intercept and slope in the linear regression both have their own
  priors. In a word, `depends_on` is unable to use within-subjects effects because each subject’s condition is
 derived from the population prior, whereas `HDDMRegressor` allows subjects to have their own intercept,
 which allows for the estimation of within-subjects variation across conditions. Thus, the choice of model
 definition is relevant to the assumptions made about the relationship between parameters and the
 experimental conditions. For more details, see Wiecki et al. (2013).

## Page 13

Advances in Methods and Practices in Psychological Science 8(1)                                             13

<Code Block 2>                                     The Gelman-Rubin statistics ( ˆR), and effective sample
```Python                                                   size (ESS) provide quantitative measures (see Box 1).
# define a model by hddm.HDDMRegressor          `az.rhat()`computes ˆR, which should be close to
m2 = hddm.HDDMRegressor(                         1 for good convergence; values below 1.01 are typically
    df, “v ~ C(conf, Treatment(‘LC’))”,       recommended (Gelman & Rubin, 1992).
    group_only_regressors = False,              `az.ess()` calculates ESS, a measure of the preci-
    keep_regressor_trace = True,                  sion of posterior estimates. If the ESS-bulk is over 400
    include=[‘a’, ‘v’, ‘t’, ‘z’, ‘sv’,          (see Box 1), the distribution’s center is well resolved,
      ‘st’, ‘sz’])                              and we should ensure high ESS across all regions of the
# fitting model and return InferenceData       parameter space (Martin et al., 2024; Vehtari et al., 2021).
m2_infdata = m2.sample(                          The latter two methods are covered by ArviZ’s `az.
    10000, chains = 4,                        summary()` (Fig. 5b).
    save_name = ‘m2’, return_infdata = True,
    sample_prior = True, loglike = True,                                      Model comparison
      ppc = True)
```                                        Upon verifying chain convergence, we proceed with
                                                model comparison to identify the best-fitting model. The
                                                           evaluation metric provided in the original HDDM is devi-  To accurately estimate parameters and ensure con-
                                                    ance information criterion (Spiegelhalter et al., 2002). Wevergence in hierarchical modeling, we set up four MCMC
                                                        include two more methods in dockerHDDM: widelychains of 10,000 samples with 5,000 burn-ins (i.e., a total
                                                           applicable information criterion (WAIC; Watanabe, 2010)of 20,000 samples for each parameter). For the more
                                                  and Pareto-smoothed importance sampling leave-one-outdetailed settings and arguments description, see “Novel
                                                           cross-validation (PSIS-LOO-CV; Vehtari et  al., 2017).Features of dockerHDDM.” With the new functionality
                                                    These methods comprehensively integrate posterior sam-introduced by dockerHDDM, we can calculate the log-
                                                            ples for model comparison and evaluation (see Box 4).likelihood of the model and generate posterior predic-
                                                         For the demonstration, we compared three modelstions after model fitting. Furthermore, the output of the
                                                          across all three evaluation metrics (lower value is bet-model fitting can be converted into InferenceData, `m2_
                                                                          ter).5 As shown in Table 5, Model 2 exhibits the lowestinfdata`, for subsequent analyses, as described in
                                                        values on  all three metrics, indicating  it  is the best“Novel Features of dockerHDDM.”
                                                    model. The results of model comparison revealed that  We emphasize that model fitting is demanding in
                                                  Models 1 and 2 are much better than the baseline Modelterms of computational resources and memory. For
                                                                    0, suggesting that experimental conflict conditions haveexample, in our tests with the Apple M1 chip, Intel
                                                        a substantial effect on drift rates. Moreover, Model 2 isi7-10700 CPU, and AMD Ryzen 9-5900HX, model fitting
                                                                   slightly better than Model 1, suggesting that regressiontook around 2 hr to 3 hr for 10,000 samples. Conse-
                                                model may suit the data better. Nevertheless, the simi-quently, fitting three models took about 6 hr to 9 hr, and
                                                                         larities between Model 1 and Model 2 suggest that bothmemory usage ranged between 8 GB and 12 GB. In
                                                  models fit the data adequately in this case.addition, if pointwise likelihood calculations (i.e., with
                                                     Note that WAIC and PSIS-LOO-CV require the point-the argument `loglike=True`) and posterior predic-
                                                     wise log-likelihood of each data point given a posteriortive data generation (i.e., with the argument `ppc=True`)
                                                   sample of parameters, which must be computed usingare enabled, an extra 1 hr to 3 hr are needed for each
                                                          the likelihood function and posterior trace (see Box 1).model. More important, the memory consumption could
                                                        This variable  is not directly provided in the HDDMescalate to 20 GB to 30 GB because pointwise likelihood
                                                           object and must be customized to be calculated via theand posterior predictive data generation will result in a
                                                           likelihood function and posterior trace.large number of new data. See discussion for recom-
                                                             In dockerHDDM, the pointwise log-likelihood canmendations to improve efficiency.
                                                 be computed at the sampling and fitting stage, via `m.
                                                    sample(. . . , retutn_infdata = True, loglike =
                                              True)` (see <Code Block 2>), or after the model hasModel diagnosis                                                   been sampled and fitted, by `m.to_infdata(loglike =
In Bayesian inference, it is crucial to ensure the conver-   True)`. Both ways return InferenceData, allowing users
gence of MCMC chains. With ArivZ, dockerHDDM sup-    to immediately compute WAIC and PSIS-LOO-CV. After
ports both visual inspection and quantitative convergence     that, the evaluation metrics for each model’s Inference-
checks (Martin et al., 2024, Chapter 10).                 Data are available using ArviZ’s `compare` method (see
  `az.plot_trace()` can be used to visualize the   <Code Block 3>), which returns the results of WAIC for
posterior distributions of parameters (i.e., trace plots of    the argument `ic=“waic”`  or PSIS-LOO-CV  for
the MCMC, Fig. 5a).                                `ic=“loo”`.

## Page 14

14                                                                               Pan et al.

   a





  b





     Fig. 5.  Model diagnosis. (a) Visualization of the traces of all chains using `az.plot_trace()`, with the argument `var_names` set
     to focus on the parameter “v_Intercept” as an example. `compact=False` and `legend=True` ensured that the individual traces of
    each chain would be visible. The Markov chain Monte Carlo (MCMC) chains are valid and reliable when they fluctuate around a value
    and different chains are indistinguishable from each other, a scenario often referred to as a “caterpillar” shape. (b) Output of `az.
    summary()`, which includes the mean and standard deviation of the Monte Carlo standard error (MCSE), the effective sample sizes
     (bulk-ESS and tail-ESS), and ˆR. Note that the summary data frame has been sorted by ˆR so that we can easily compare the minimum
    and maximum values of ˆR.

## Page 15

Advances in Methods and Practices in Psychological Science 8(1)                                             15


Box 4.  Linking Deviance Information Criterion, Widely Applicable Information Criterion, and Pareto-Smoothed
Importance Sampling Leave-One-Out Cross-Validation to Akaike Information Criterion

 The deviance information criterion (DIC), widely applicable information criterion (WAIC), and Pareto-
 smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV) are criteria founded on the
 concept of out-of-sample predictive accuracy, that is, the accuracy of using the fitted model to predict new
 data generated by the assumed data-generating process. Predictive accuracy is often encapsulated by the log
 predictive density (Box 1). However, the log predictive density approximated using the observed data and the
 posterior estimates of parameters is biased, and an adjustment is required to correct the bias. Thus, the key
 difference between DIC, WAIC, and PSIS-LOO-CV lies in the difference between the two terms of log
 predicted density and corrected bias (see the table below).

                               
                                         Predictive accuracy     Adjustment            Formula
                  AIC                  log p ( y | ˆθmle )           k          −2 ( log p ( y | θmleˆ   ) −k )
                 DIC                 log p ( y | θBayes )           PDIC        −2 (                                                                                         log p ( y | θBayes ) − PDIC )
               WAIC                    lpd                     ˆpWAIC           −2 ( lpd − pWAICˆ     )
                  PSIS-LOO-CV         elpd psis − loo                          −2elpd psis − loo
                       Note: lpd = computed log pointwise predictive density, see Glossary for details; elpd psis − loo =
                      expected log pointwise predictive density for a new dataset based on PSIS-LOO method; k = the
                      count of model parameters; PDIC = the DIC’s adjustment for the effective number of parameters
                          (Spiegelhalter et al., 2002); ˆpWAIC = the WAIC’s approach to adjusting the effective number of
                       parameters (Watanabe, 2010). DIC = deviance information criterion; WAIC = widely applicable
                        information criterion; PSIS-LOO-CV = Pareto-smoothed importance sampling leave-one-out cross-
                          validation.


   DIC uses the Bayesian posterior means for estimating log predictive density and includes an adjustment
 based on the effective number of parameters (PDIC). It is particularly suited for hierarchical models, offering an
 improved estimate of predictive density (Spiegelhalter et al., 2002).
   WAIC further refines DIC, evaluating the log predictive density across the entire posterior and correcting
 bias via the variability of log predictive density ( ˆpWAIC ). This adjustment is crucial for measuring model
 robustness and guarding against overfitting (Watanabe, 2010). Both DIC and WAIC rely on estimating the
 effective number of parameters, but DIC assumes a Gaussian distribution for the likelihood, which simplifies
 the calculation (Lunn et al., 2012). In contrast, WAIC does not rely on this strict assumption and uses the full
 posterior distribution, offering greater flexibility and accuracy but at a higher computational complexity
 (Gelman et al., 2014).
   PSIS-LOO-CV estimates the predictive density by simulating the leave-one-out cross-validation, which by
 definition is the out-of-sample predictive accuracy, so bias correction is no longer needed for PSIS-LOO-CV.
 For more details on these three indices, see Gelman et al. (2014) and Vehtari et al. (2017).


<Code Block 3>                                            the PPC, as discussed in the next section, alongside the
```Python                                                  diagnostic information provided by LOO and WAIC (see
compare_dict = {                                      Martin et al., 2024, Chapter 5; Vehtari et al., 2017).
    ‘m0’: m0_infdata,
    ‘m1’: m1_infdata,
                                  PPC    ‘m2’: m2_infdata
}                                                           In addition to model comparison, which assesses relative
az.compare(compare_dict, ic = ‘loo’)           performance, the PPC evaluates how well predictive data
```                                                   generated from posterior samples of parameters align
                                                       with the actual data. PPC is crucial because model com-
   Finally, we note that the model-comparison metrics allow    parison evaluates only the “least worst” model, but this
only a relative ranking of alternatives. To assess the absolute   model may not necessarily account for the data very well
goodness of fit of the model, we recommend performing    (see Martin et al., 2024, Chapter 5).

## Page 16

16                                                                               Pan et al.


Table 5.  Model Comparison With Different Criteria            synthetic data from Model 2 match more closely the
                                                             actual data compared with the baseline Model 0, and
Ranka        DIC         PSIS-LOO-CV       WAIC
                                                                    this difference becomes apparent when examining PPC
1      m2 (10,654.89)   m2 (10,646.25)   m2 (10,646.20)     at the individual level (Fig. 6a) and condition level (Fig.
2      m1 (10,655.24)   m1 (10,647.21)   m1 (10,647.15)    6b). Other approaches for PPCs can be used to quantify
3      m0 (10,835.24)   m0 (10,824.93)   m0 (10,824.89)    accordance between data and model across quantiles of
                                                          the response time (RT) distribution, for example, usingNote: DIC = deviance information criterion; PSIS-LOO-CV = Pareto-
smoothed importance sampling leave-one-out cross-validation; WAIC      Bayesian predictive versions of quantile probability plots
widely applicable information criterion; m0 = Model 0; m1 = Model 1;     (Frank et al., 2015; Ging-Jehli et al., 2021), and example
m2 = Model 2.                                        code in HDDM is available on request.
aRank is ranging from the best model to the worst.

  ArviZ offers convenient visualization tools for inspect-                                                  Statistical inferenceing PPC (Kumar et al., 2019). The function `az.plot_
ppc()` is helpful to visualize PPC at the individual or   A final step in Bayesian modeling is to draw statistical
condition level (Fig. 6). In the demonstration, the    inferences from the posterior parameter distributions in
  a





        −5     0     5    10  −15  −10  −5    0    5   10       −5     0     5     10  −10  −5    0    5   10
                         rt / rt                                     rt / rt                                       rt / rt                                     rt / rt
                3                        11                         3                        11
  b





    −10   −5    0    5    10  −15 −10  −5   0   5   10      −10  −5    0    5    10        −5     0     5    10
                         rt / rt                                     rt / rt                                       rt / rt                                     rt / rt
               LC                     HC                        LC                     HC

                                           Posterior Predictive           Observed             Posterior Predictive Mean

    Fig. 6.  Posterior predictive check plot `az.plot_ppc()` for Model 0 “m0” and Model 2 “m2.” Solid black lines are the density plot of
    the observed response time (RT) data; blue lines are the posterior predictive samples; each line represents the predicted RT distribution
   based on one posterior predictive sample; yellow dashed lines represent the mean of all predicted RT distributions across all posterior
    predictive samples. (a) Results of the comparison between the two models (m0 vs. m2) at the individual level (Subjects 3 and 11 as an
   example). (b) Results of the comparison at the condition level (i.e., “LC” represents lower conflict, and “HC” represents higher conflict).
    All plots in the left column are for m0, and all plots in the right column are for m2. Note that the argument `coords` specifies the
    posterior-predictive-check level (individual or group level) that should be preprocessed before plotting. `num_pp_samples` is used to
    set the number of predictive data required for plotting.

## Page 17

Advances in Methods and Practices in Psychological Science 8(1)                                             17

    a                             b





                          v_C(conf, Treatment('LC'))[T.HC]                                  v_LC                v_HC                                                                             1.50

                                                                             1.25
                 mean == −−0.54
                                                                             1.00

                                                                             0.75

                                                                             0.50
                  0.0%% inin ROROPE
                                                                             0.25
                  95%95% HDHDI
                                                                             0.00
                                      −0.2             0.2
                                                                       −0.25
               −0.67            −0.42
                                                                       −0.50
                    0.8     0.6     0.4    0.2     0.0     0.2

    Fig. 7.  (a) Statistical inference of parameters. The high-density interval (HDI; black line and texts) is compared with the region of
     practical equivalence (ROPE; red line and text). `var_names` argument can be used to select both group-level and individual-level
    parameters for analysis. `hdi_prob` argument specifies the probability of the HDI, typically set at 0.95 to correspond to a 95% credible
     interval. `rope` defines the limitations of ROPE, which is a range considered to be equivalent to the null hypothesis or a reference
    value for the parameter. The results show no overlap between the 95% HDI and the ROPE, indicating that the parameter is credibly
     different from zero. (b) Violin plot of parameter posteriors at two conflict levels. The black line is the 95% HDI, and the white dot is
    the mean. The drift rate is lower in high-conflict (HC) conditions than in low-conflict (LC) conditions.




the best-fitting model. In our example, we test the    equivalence (ROPE; Kruschke, 2018; see Box 1). In addi-
hypothesis of whether drift rates significantly differ    tion, we provide methods for calculating BFs in the
between HC and LC conditions based on Model2 (“m2”   Appendix.
in the Notebook). This hypothesis is tested using the    We define a ROPE of [–0.2, 0.2] to represent values
posterior samples of the regression coefficient in “m2,”    practically equivalent to zero6 and use the `plot_pos-
which has a variable name “v_C(conf, Treatment(‘LC’))   terior()` function from ArviZ to implement the ROPE
[T.HC]”.                                                                 test. By comparing the 95% HDI of the regression coef-
  Note that there are several acceptable methods for     ficient to this ROPE, we find that the HDI falls com-
Bayesian hypothesis testing, such as BFs (Boehm et al.,    pletely outside the ROPE (Fig. 7a), suggesting that the
2023; Wagenmakers et al., 2010), maximum a posteriori     drift rate is higher in the LC condition than the HC
based p value (Mills, 2018), directional probabilities    condition (Fig. 7b).
(Makowski et al., 2019), and the full Bayesian signifi-       Therefore, considering the results from various aspects
cance test (Kelter, 2022). In cognitive science and psy-   (model comparison, PPC, and posterior inference),
chology, although BFs are often advocated as a Bayesian   we conclude that the model that takes into account
alternative to frequentist p values (Kelter, 2021; van de    the influence of conflict level on drift rate performs the
Schoot et al., 2017; Wagenmakers et al., 2010), debate    best. Moreover, HC affects the cognitive process of
remains about which Bayesian measures should be used    decision-making by impeding the speed of evidence
in which settings of scientific hypothesis testing (Kelter,    accumulation.
2023; Makowski et al., 2019). Therefore, it is useful to
consider various Bayesian hypothesis-testing methods                                           Discussiondepending on the study objectives and design (Kelter,
2023; Kruschke, 2021; Makowski et al., 2019).             In this tutorial, we focus on an easy-to-use computa-
  Here, we demonstrate Bayesian inference using    tional environment for HDDM, including installation of
an approach that combines the approach combining    the tool, its features, and case applications. Although
highest density interval (HDI) and the region of practical   some conceptual discussions have been addressed in

## Page 18

18                                                                               Pan et al.


  Table 6.  Tools Comparison for Modeling Hierarchical DDM

                           (docker)HDDM       brms/RStan/hBayesDM        JAGS             EMC2

   Language                Python                       R                 R                 R
  MCMC Algorithm          Metropolis-Hastings     NUTS                   Gibbs sampling      Particle Metropolis
   Support models        DDM, full DDM,        DDM, full DDM       DDM           DDM, LBA, RDM, etc.
                      RLDDM, collapsing
                          boundary variants, etc.
  Custom prior                 No                    Yes                 Yes                 Yes
   Linear mixed extension            Yes                    Yes                 Yes                 Yes
   Likelihood-free                   Yes                 No               No              No

   Note: DDM = drift-diffusion model; MCMC = Markov chain Monte Carlo; RLDDM = reinforcement learning drift diffusion model; LBA = linear
    ballistic accumulator; RDM = racing diffusion model.



other articles (Boag et al., 2024; Shinn et al., 2020; Voss    (Matzke & Wagenmakers, 2009) and applicable to typical
et  al., 2013), we nevertheless discuss some relevant    cognitive experiments.
issues below.                                           Another advantage of HDDM is its support for diverse
                                                      accumulation models, including models with collapsing
                                                    boundaries and those integrated with reinforcementWhy use dockerHDDM among tools?                                                             learning, called “RLDDM” (Fengler et al., 2022; Pedersen
Inference for the DDM can be implemented via multiple   & Frank, 2020; Pedersen et al., 2017). In addition, the
software/packages, such as fast-DM (Voss & Voss, 2007),     latest version of HDDM provides many likelihood-free
flexDDM (LaFollette et al., 2024), rtdists (Singmann et al.,    models, broadening its applications. For instance, its
2022), EZ-DDM (Wagenmakers et al., 2007), and pyDDM    integration with neural networks, such as the LANs (like-
(Shinn et al., 2020). For more details on tool and algorithm    lihood approximation networks; Fengler et al., 2021),
comparisons, see Shinn et al. (2020). Although all the    has greatly enhanced the efficiency of model design and
above tools are estimated in a frequency framework and    development.
fit data at the individual-participant level, HDDM takes the     A notable limitation of dockerHDDM is its lack of
Bayesian approach and estimates model parameters at    integration with the most advanced parameter-estimation
both the individual and group levels (i.e., the hierarchical-    techniques. For instance, its successors, HSSM and EMC2,
model or multilevel-model approach; see Wiecki et al.,   have begun incorporating advanced MCMC methods.
2013). Tools that also allow the Bayesian hierarchical    Moreover, innovative neural-network approaches, such
modeling approach of DDM include brms based on RStan    as LANs (Fengler et al., 2021), MNLE (Boelts et al., 2022),
(Henrich et  al., 2023), the Wiener module in JAGS   and Bayesflow (Radev et al., 2022), have the potential
(Wabersich & Vandekerckhove, 2014), EMC2 (Stevenson    to significantly enhance these estimation procedures.
et al., 2024), and hBayesDM (Ahn et al., 2017). For com-   However, the mastery of these cutting-edge techniques
parison between these tools and HDDM, see Table 6.      requires a higher level of expertise to prevent misuse.
  HDDM stands out for its ease of use, enabling users      Consequently, we propose that the mission of dock-
to construct and fit basic models with just a few lines of   erHDDM should be to streamline operations and lower
code.  It facilitates the definition of complex mixed-    the barrier to entry, facilitating analogical learning and,
effects models without the need for prior specifications,    ultimately, preparing users for the transition to the more
making it more accessible for beginners. Although brms    sophisticated methods.
and EMC2 also define mixed-effects models well, they
necessitate users to manually define prior distributions   Whether to include parameters’
for random effects and covariance structures. In addition,                                                 intertrial variability?RStan and JAGS require expertise in linear model repa-
rameterization. The absence of this expertise may result   As a demonstration, we used the seven-parameter full
in model-fitting failures or biased estimates. On the other   DDM. If a user wishes to fit only the four-parameter model,
hand, the simplicity of HDDM comes at the cost of flex-    the unnecessary parameters can be removed from the
ibility because it restricts users to the default priors (see    include argument, for example, `include=[‘a’, ‘v’,
Box 3) and does not allow for customization. However,   ‘t’, ‘z’]`. In contrast, the full model, which integrates
the weakly informative prior implemented in HDDM was     trial-by-trial variability, is known for its robustness in fitting
based on previous meta-analyses of published results    various data sets and accommodating extreme response

## Page 19

Advances in Methods and Practices in Psychological Science 8(1)                                             19


times, including fast and slow errors (Schubert et al., 2017).    2024). For further empirical guidelines, see Boehm et al.
However, Lerche and Voss (2016) argued that excluding    (2018) and Lerche and Voss (2017).
trial-by-trial parameters can enhance the fit and recovery      Note that parameter estimation can be affected by
of fundamental parameters.                             extreme values, such as very fast response times. HDDM
  Consequently, the choice to include trial-by-trial vari-    addresses this issue by assuming a mixture model in
ability requires a delicate balance between the predic-   which a proportion of the response times are from a
tion and complexity of the model and the specific    uniform distribution (Ratcliff & Tuerlinckx, 2002; Wiecki
requirements of the data. Given the extensive data     et al., 2013). The proportion of response times is con-
requirements for inferring across-trial variability, our    trolled by the parameter `p_outlier`, which is set to
stance is to cautiously include across-trial variability in    0.05 by default. This approach helps mitigate the effect
the model for a more robust fit and more precise infer-    of extreme values and ensures a more robust parameter
ence of the basic parameters (see similar discussion in    estimation.
Boag et al., 2024). For instance, because the variability        Finally, it is essential to conduct PPCs to validate the
of the nondecision time tends to be easily recovered   model (see “PPC”). These checks help to ensure that the
(e.g., the result of the parameter recovery in Appendix   model is capable of accurately reproducing the observed
Figure S2), it may be prudent to include only this param-    data, thus providing confidence in the evaluation of the
eter but not the other variability parameters by default.   model and parameters.
Nevertheless, when the data set is substantial and the
research objective prioritizes the analysis of specific
                                        Computational resources and tipsresponse-time patterns, such as fast or slow errors, the
selective integration (the parameter variability of drift   To achieve accurate estimates, more subjects, more trials,
and start point; also see Table 1) of these parameters   and often more samples are required, leading  to
may be warranted. We recommend reading the work by    increased demands for computational resources. This is
Boehm et al. (2018), which offers expert advice and    not unique to dockerHDDM; other tools using MCMC
recommendations on estimating across-trial variability    algorithms, such as DMC and brms mentioned earlier,
parameters.                                               are also affected by these factors. In the examples pro-
                                                      vided in this article, fitting each model with 14 subjects
Data quantity and quality for fitting         and 3,988 trials takes 2 hr to 3 hr and requires 8 GB to
                                                    12 GB of memory. Running out of memory can causethe DDM                                                          the Jupyter kernel to suspend and restart, interrupting
Both the number of subjects and the number of trials    the  process.  Predictably, computational resources
should be considered. Because of the hierarchical nature   become a limiting factor with increasing data. To facili-
of the model, hierarchical models typically require fewer    tate better model analysis, we offer the following tips
trials than nonhierarchical models (Alexandrowicz &   and recommendations.
Gula, 2020; Wiecki et al., 2013). In general, 12 subjects
are sufficient to obtain stable results (Wiecki et al., 2013),    Initial testing.  When initially building the model, use
but we recommend collecting data from more than 20    subset data from a small number of subjects and reduce
subjects for a more robust fit. However, the number of    the MCMC sample size to verify that the model definition
sufficient trials varies depending on the parameters of   and code are correct. Once validated, increase the data
interest. For the basic four-parameter model, the number   and sample sizes.
of  trials has a small effect on parameter estimates
(Alexandrowicz & Gula, 2020). Twenty trials appears to   Adjust memory settings.  If users experience a Jupyter
be the minimum standard, and more than 50 trials tend    kernel suspension or  restart because of memory con-
to produce robust results (Wiecki et al., 2013). Estimates     straints, they can attempt to configure or increase virtual
of t and z tend to be superior to those of a and v. To   memory. For Windows users, it is necessary to check and
obtain more accurate estimates of v, a number of trials   remove the memory-usage limitations imposed by WSL
greater than 100 is recommended (Alexandrowicz &   (Windows Subsystem for Linux).
Gula, 2020). For parameters such as sv, st, and sz, a large
number of trials are required for estimation, preferably   Separate execution.  Model fitting, calculation of point-
more than 120 trials (Wiecki et al., 2013). Recent dis-    wise log likelihood, and generation of PPCs data can be
course has emphasized that the determination of the    executed separately. This approach helps prevent inter-
number of subjects and trials should be aligned with    rupting long-running processes because of errors and
considerations of experimental design, desired target    ensures that each step can be independently validated and
effects, and parameter recovery simulations (Boag et al.,   debugged before proceeding to the next.

## Page 20

20                                                                               Pan et al.


Box 5.  Recommendation for Further Reading

 A full understanding of how Bayesian hierarchical drift-diffusion modeling works requires not only basic
  knowledge of drift-diffusion modeling but also knowledge of Python programming, Bayesian statistics, and
  hierarchical regression models. This background knowledge is generally not part of the coursework in
  psychology or neuroscience education, although the situation has been changing in recent years. We
  recommend the following resources to quickly catch up and avoid misuse or abuse of hierarchical drift-
  diffusion modeling.


   Background knowledge/skills                                             Resource

   Bayesian statistics                           Etz & Vandekerckhove, 2018; Kruschke, 2014, 2018; Lambert, 2018; Martin
                                                        et al., 2024; McElreath, 2020; van de Schoot et al., 2021.
   (Bayesian) Hierarchical (regression)         https://twiecki.io/blog/2014/03/17/bayesian-glms-3/; https://github.com/lei-
     models                                zhang/BayesCog_Wien;
                                              Capretto et al., 2020.
   Computational modeling                Blohm et al., 2020; Busemeyer, 2015; Busemeyer & Diederich, 2009; Etz &
                                            Vandekerckhove, 2018; Farrell & Lewandowsky, 2018; Lee & Wagenmakers,
                                                 2014; Wilson & Collins, 2019; Zhang et al., 2020.
   Drift-diffusion models                  Boag et al., 2024; Ratcliff et al., 2016; Ratcliff & McKoon, 2008; Voss et al.,
                                                 2013.
   Sequential-sampling models beyond         Fengler et al., 2022; Forstmann et al., 2016; Ratcliff et al., 2016.
      drift-diffusion models



Notebook segmentation.  Fit models into separate note-    modeling. Given the extensive knowledge required for
books to reduce the resource load of loading multiple    principled computational modeling, we recommend
models.                                                 readers refer to the materials in Box 5 for a deeper
                                                       understanding of the DDM family, computational model-
Model saving.  Save the fitted models and then load only    ing, hierarchical models, and Bayesian modeling. We
the InferenceData  files instead of the entire models to    expect that dockerHDDM and this detailed tutorial will
reduce resource usage.                                 reduce the technical burden and help readers get started
                                                       with computational modeling. Ultimately, we hope that
Cloud deployment.  Docker is easily deployed in cloud-     this tool and the computational-modeling concepts pre-
computing environments (or use the docker image in Sin-    sented in the tutorial will promote the computational
gularity). Use your institution’s computing services or rent    reproducibility of drift-diffusion modeling for users of
cloud computing services to handle larger data sets.            all levels of computational expertise.

Summary                              Appendix
In this  article, we introduce dockerHDDM, a user-   Bayesian hypothesis testing with
friendly, out-of-the-box, and one-stop Docker image for                                         Savage–Dickey methodimplementing HDDM analysis within a modern Bayesian
hierarchical workflow. Our dockerHDDM has three   Another method to test the experimental effect is to
major advantages: (a) It leverages Docker to solve com-   compute the Savage-Dickey density ratio to approximate
patibility issues and simplify the installation process, (b)    the Bayes factor (see Box 1). ArviZ provides the `plot_
it ensures broad support across different machines   bf` function to visualize the differences between prior
equipped with either Intel or Apple chips, and (c)  it   and posterior distributions and compute the Bayes fac-
integrates state-of-the-art Bayesian modeling practices     tor. Note that the Savage-Dickey ratio is related to the
with ArviZ, facilitating a more principled Bayesian work-     prior, which is weak in HDDM, resulting in very large
flow. We also provide a step-by-step video tutorial on    Bayes-factor values. We therefore urge caution in using
how to use dockerHDDM.                                     this method and that inference should be drawn by
  Although we have provided a step-by-step guide to   combining as many as possible (e.g., highest density
using dockerHDDM, it is unfortunately not possible to    interval or highest density interval + region of practical
provide a comprehensive introduction to computational    equivalence as mentioned in “Statistical Inference”).

## Page 21

Advances in Methods and Practices in Psychological Science 8(1)                                             21

  a                                b





        7
                                                                   Prior                                                                             Prior
                                                               Posterior           40                                                    Posterior
        6

        5                                                                  30

        4
                                                                  20        3                                                                                                                                            Density

        2                                                                  10

        1

        0                                                          0
         −1.0   −0.8   −0.6   −0.4   −0.2    0.0      0.2     0.4             0.30   0.35   0.40   0.45   0.50   0.55   0.60   0.65   0.70
                                                                                                     z                    v_C(conf, Treatment(‘LC’))[T.HC]

   Figure S1.  Bayes factor test. This figure illustrates the prior (blue line) and posterior (orange line) density distributions for the drift-rate
   parameter under the conflict condition. The dashed vertical line represents the reference/null value (zero), and the black dot indicates
    the Bayes factor at this point. The notable difference between the probabilistic density of prior and posterior distributions at the refer-
   ence value, which is used to calculate the Savage-Dickey density ratio and approximate the Bayes factor, provides evidence to accept
    or reject the experimental effect.

   In Figure S1, the left panel displays the Bayes factor   Parameter-recovery result
favoring the alternative hypothesis (BF10 = 1 . 5 × 10 236, BF01 =
0), indicating extremely strong evidence supporting the    Wiecki et al. (2013) demonstrated the superiority of
alternative hypothesis over the null hypothesis. This    Bayesian methods and hierarchical models for parameter
implies that the conflict condition significantly affects the    recovery in HDDM. We illustrate the parameter recovery
drift rate. The right panel shows the Bayes factor favoring    analysis of Model 2 in Figure S2. The results show that
the null hypothesis (BF10 = 0. 14, BF01 = 7.15), indicating    our model-fitting approach can yield good parameter
moderate evidence supporting the null hypothesis over    recovery. For the code that repeats this result, see https://
the alternative hypothesis. This suggests that there is no   github.com/hcp4715/dockerHDDM/blob/master/dock
response bias, as evidenced by z being close to 0.5.       erHDDMTutorial/Parameter_recovery.ipynb.
      a





                                                                                            Fig. S2. (continued on next page)

## Page 22

22                                                                               Pan et al.

      b





           Figure S2.  Model 2 parameter-recovery results. Blue is the true parameter, orange is the recovered parameter, white
             dots are the means, and the bar is the 95% highest density interval (HDI) range. Subplot A shows the parameter-
            recovery results at the group level, including eight parameters, of which, the first five are basic parameters and the
                last three are trial-by-trial variants. Subplot B shows the parameter-recovery results at individual level, including five
             basic parameters for 13 subjects out of 65.

## Page 23

Advances in Methods and Practices in Psychological Science 8(1)                                             23

Transparency                                                     Lei Zhang    https://orcid.org/0000-0002-9586-595X
Action Editor: Rogier Kievit                               Ru-Yuan Zhang    https://orcid.org/0000-0002-0654-715X
Editor: David A. Sbarra                             Hu Chuan-Peng    https://orcid.org/0000-0002-7503-5131
Author Contributions
  Wanke Pan: Software; Validation; Visualization; Writing –   Acknowledgment
   original draft.
                                                       Thanks to HDDM (Wiecki et al., 2013; Fengler et al., 2021;  Haiyang Geng: Conceptualization; Writing – original draft;
                                                              Fengler et al., 2022) and ArviZ (Kumar et al., 2019) for the   Writing – review & editing.
                                                     open resource. We thank Dr. Mads Lund Pedersen for his open   Lei Zhang: Conceptualization; Writing – original draft;
                                                                     dockerfile, which insipre the current project. We also thank   Writing – review & editing.
                                                                the netizens for their time in testing and valuable feedback,  Alexander Fengler: Writing – original draft; Writing –
                                                       which allows us to continuously improve the tools and tutori-   review & editing.
                                                                                 als. We appreciate the help of Dr. Yuan Rui in the early stage  Michael J. Frank: Writing – original draft; Writing – review
                                                                   of docker image development.  & editing.
  Ru-Yuan Zhang: Conceptualization; Funding acquisition;
                                                 Notes   Supervision; Writing – original draft; Writing – review &
   editing.                                                                  1. Note that `/home/jovyan/{any_folder_name}` is a path
  Hu Chuan-Peng: Conceptualization; Funding acquisition;    mounted in the Jupyter Docker image and that `{any_folder_
   Software; Supervision; Writing – original draft; Writing –    name}` will be visible in the browser. The default username is
   review & editing.                                        `jovyan`, and it cannot be changed.
Declaration of Conflicting Interests                                     2. For beginners unfamiliar with Jupyter Notebook, do not panic!
  The author(s) declared that there were no conflicts of inter-       It is just an interface where you can write code and immedi-
   est with respect to the authorship or the publication of this     ately check results. You may visit the official website at https://
    article.                                                        jupyter.org/try-jupyter/notebooks/?path=notebooks/Intro.ipynb
Funding                                                              to try out a web-based platform online. The Jupyter website also
   This work was supported by National Key R&D Program of    provides extensive documentation for users who want to learn
   China (2023YFF1204200 to R.-Y. Zhang), the National Natu-    more about Jupyter Notebook and Python programming (see
    ral Science Foundation of China (32471097 to H. Chuan-     https://docs.jupyter.org/en/latest/).
   Peng; 32441102 and 32100901 to R.-Y. Zhang), Natural     3. To run the example notebooks faster, we use only 500 samples
   Science Foundation of Shanghai (21ZR1434700 R.-Y. Zhang),     here. For a more in-depth understanding of the MCMC settings,
  and the Austrian Science Fund (FWF-M3166) to L. Zhang.    we recommend reading van de Schoot et al. (2017); and Wiecki
Open Practices                                                           et al. (2013). The burn-in samples serve to calibrate the fitting,
   All resources are available on OSF  at  https://osf.io/    so the final samples need to exclude burn-in samples, yielding a
   3upng/?view_only=2425347775e749c3bab67af68607b918,     total of 500 – 100 = 400 samples per chain. Generally, a larger
  which is linked to the GitHub repository at https://github    number of samples improves the estimation accuracy of a model.
  .com/hcp4715/dockerHDDM/ and other resources. The soft-     4. InferenceData is a more modern data construct that contains
   ware, data, and scripts (Jupyter notebooks) used to generate     prior, posterior, and a posterior predictive samples and observed
   the models and results described in this article can be     data, facilitating the visualization and analysis of multiple joint
   accessed via the dockerHDDM image at https://hub.docker    data sets (Hoyer & Hamman, 2017; Kumar et al., 2019).
   .com/r/hcp4715/hddm. Alternatively, readers can find our     5. Deviance information criterion can be extracted directly from
   online notebooks and related materials at https://github     the model rather than InferenceData, for example, `m0.dic`.
  .com/hcp4715/dockerHDDM/ and https://github.com/     6. The ROPE should be tailored to the specific paradigm and
   hcp4715/dockerHDDM/tree/master/OfficialTutorials. In    research question (Dienes, 2021) and reflect the range of pos-
   addition, the code used to create our dockerHDDM images     sible values for each parameter (e.g., Tran et  al., 2021). For
    is available at https://github.com/hcp4715/dockerHDDM/    example, a recent systematic parameter review of DDM found
   blob/master/Dockerfile. For any questions regarding this     that the absolute value of a drift rate ranged from 0.01 to 18.51,
   tutorial or related dockerHDDM images, discussions can be    with a median of 2.25 (Tran et al., 2021); another simulation and
   held at https://github.com/hcp4715/dockerHDDM/discus     meta-analysis of conflict tasks showed that a drift rate between
   sions. This article has received the badges for Open Data     0.05 and 0.35 captured the conflict effect (Hedge et al., 2018).
  and Open Materials. More information about the Open Prac-    Thus, we choose ROPE [–0.2, 0.2] for illustrative purposes, imply-
   tices badges can be found  at http://www.psychologi     ing that effects on drift rates smaller than 0.2 are not of interest.
   calscience.org/publications/badges.
                                                   References
                                                          Ahn, W.-Y., Haines, N., & Zhang, L. (2017). Revealing neurocom-
                                                                       putational mechanisms of reinforcement learning and deci-
                                                               sion-making with the hBayesDM package. ComputationalORCID iDs
                                                                        Psychiatry, 1, 24–57. https://doi.org/10.1162/cpsy_a_00002
Wanke Pan    https://orcid.org/0000-0002-0896-6833           Alexandrowicz, R. W., & Gula, B. (2020). Comparing eight
Haiyang Geng    https://orcid.org/0000-0001-6115-807X           parameter estimation methods for the Ratcliff diffusion

## Page 24

24                                                                               Pan et al.


   model using free software. Frontiers in Psychology, 11,    Donkin, C., & Brown, S. D. (2018). Response times and
    Article 484737. https://doi.org/10.3389/fpsyg.2020.484737        decision-making. In  J. T. Wixted (Ed.), Stevens’ hand-
Annis, J., Miller, B. J., & Palmeri, T. J. (2017). Bayesian infer-       book of experimental psychology and cognitive neurosci-
   ence with stan: A tutorial on adding custom distributions.       ence (pp. 1–33). John Wiley & Sons. https://doi.org/10
    Behavior Research Methods, 49(3), 863–886. https://doi.org/        .1002/9781119170174.epcn509
    10.3758/s13428-016-0746-9                                       Etz, A., Chávez de la Peña, A. F., Baroja, L., Medriano, K.,
Blohm, G., Kording, K. P., & Schrater, P. R. (2020). A how-     & Vandekerckhove, J. (2024). The HDI + ROPE decision
   to-model guide for neuroscience. Eneuro, 7(1), Article         rule is logically incoherent but we can fix it. Psychological
   ENEURO.352-19.2019. https://doi.org/10.1523/ENEURO        Methods. Advance online publication. https://doi.org/10
    .0352-19.2019                                               .1037/met0000660
Boag, R. J., Innes, R., Stevenson, N., Bahg, G., Busemeyer, J. R.,     Etz, A., & Vandekerckhove, J. (2018). Introduction to Bayesian
    Cox, G. E., Donkin, C., Frank, M., Hawkins, G., Heathcote, A.,        inference for psychology. Psychonomic Bulletin & Review,
   Hedge, C., Lerche, V., Lilburn, S., Logan, G. D., Matzke, D.,         25(1), 5–34. https://doi.org/10.3758/s13423-017-1262-3
    Miletic,  S., Osth, A.  F., Palmeri, T., Sederberg, P. B.,    Evans, N. J., & Wagenmakers, E.-J. (2020). Evidence accumula-
      . . . Forstmann, B. (2024). An expert guide to planning         tion models: Current limitations and future directions. The
    experimental tasks for evidence accumulation modelling.        Quantitative Methods for Psychology, 16(2), 73–90. https://
    PsyArXiv. https://doi.org/10.31234/osf.io/snqgp                 doi.org/10.20982/tqmp.16.2.p073
Boehm, U., Annis, J., Frank, M. J., Hawkins, G. E., Heathcote, A.,     Farrell, S., & Lewandowsky, S. (2018). Computational model-
    Kellen, D., Krypotos, A. M., Lerche, V., Logan, G. D., Palmeri,        ing of cognition and behavior. Cambridge University Press.
    T.  J., van Ravenzwaaij, D., Servant, M., Singmann, H.,        https://doi.org/10.1017/CBO9781316272503
    Starns,  J.  J., Voss,  A., Wiecki, T.  V., Matzke, D., &    Fengler, A., Bera, K., Pedersen, M. L., & Frank, M. J. (2022).
   Wagenmakers, E. J. (2018). Estimating across-trial variability       Beyond drift diffusion models: Fitting a broad class of
    parameters of the Diffusion Decision Model: Expert advice        decision and reinforcement learning models with HDDM.
   and recommendations. Journal of Mathematical Psychology,        Journal of Cognitive Neuroscience, 34(10), 1780–1805.
    87, 46–75. https://doi.org/10.1016/j.jmp.2018.09.004             https://doi.org/10.1162/jocn_a_01902
Boehm,  U., Evans, N.  J., Gronau, Q.  F., Matzke,  D.,    Fengler, A., Govindarajan, L. N., Chen, T., & Frank, M.  J.
   Wagenmakers, E.-J., & Heathcote, A. J. (2023). Inclusion        (2021). Likelihood approximation networks (LANs) for
   Bayes factors for mixed hierarchical diffusion decision          fast inference of simulation models in cognitive neurosci-
    models. Psychological Methods, 29, 625–655. https://doi        ence. eLife, 10, Article e65074. https://doi.org/10.7554/
    .org/10.1037/met0000582                                        eLife.65074
Boelts, J., Lueckmann, J.-M., Gao, R., & Macke, J. H. (2022).    Forstmann, B. U., Ratcliff, R., & Wagenmakers, E.-J. (2016).
    Flexible and efficient simulation-based inference for mod-        Sequential sampling models in cognitive neuroscience:
    els of decision-making. eLife, 11, Article e77220. https://        Advantages, applications, and extensions. Annual Review
    doi.org/10.7554/eLife.77220                                         of Psychology, 67(1), 641–666. https://doi.org/10.1146/
Busemeyer, J. R. (Ed.). (2015). The Oxford handbook of compu-        annurev-psych-122414-033645
    tational and mathematical psychology. Oxford University    Frank, M. J., Gagne, C., Nyhus, E., Masters, S., Wiecki, T. V.,
    Press.                                                     Cavanagh, J. F., & Badre, D. (2015). fMRI and EEG predic-
Busemeyer, J. R., & Diederich, A. (2009). Cognitive modeling.         tors of dynamic decision parameters during human rein-
    Sage.                                                      forcement learning. The Journal of Neuroscience, 35(2),
Capretto, T., Piho, C., Kumar, R., Westfall, J., Yarkoni, T., &        485–494. https://doi.org/10.1523/JNEUROSCI.2036-14.2015
    Martin, O. A. (2020). Bambi: A simple interface for fitting    Gelman, A., Hwang, J., & Vehtari, A. (2014). Understanding pre-
   Bayesian linear models in python. arXiv. https://doi.org/         dictive information criteria for Bayesian models. Statistics
    10.48550/ARXIV.2012.10754                            and Computing,  24(6), 997–1016.  https://doi.org/10
Cavanagh, J. F., Wiecki, T. V., Cohen, M. X., Figueroa, C. M.,        .1007/s11222-013-9416-2
   Samanta,  J., Sherman,  S.  J., & Frank, M.  J. (2011).    Gelman, A., & Rubin, D. B. (1992). Inference from iterative
    Subthalamic nucleus stimulation reverses mediofrontal        simulation using multiple sequences. Statistical Science,
    influence over decision threshold. Nature Neuroscience,         7(4), 457–472.
    14(11), 1462–1467. https://doi.org/10.1038/nn.2925         Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C.,
Chandrasekaran, C., Peixoto, D., Newsome, W. T., & Shenoy,        Carpenter, B., Yao, Y., Kennedy, L., Gabry,  J., Bürkner,
    K. V. (2017). Laminar differences in decision-related neural         P.-C., & Modrák, M. (2020). Bayesian workflow. arXiv.
     activity in dorsal premotor cortex. Nature Communications,        https://doi.org/10.48550/arXiv.2011.01808
    8(1), Article 614. https://doi.org/10.1038/s41467-017-00715-0     Ging-Jehli, N. R., Ratcliff, R., & Arnold, L. E. (2021). Improving
Desai, N., & Krajbich, I. (2022). Decomposing preferences into        neurocognitive testing using computational psychiatry—
    predispositions and evaluations. Journal of Experimental      A systematic review for ADHD. Psychological Bulletin,
    Psychology-General, 151(8), 1883–1903. https://doi.org/        147(2), 169–231. https://doi.org/10.1037/bul0000319
    10.1037/xge0001162                                   Hedge, C., Powell, G., Bompas, A., Vivian-Griffiths, S., &
Dienes, Z. (2021). Obtaining evidence for no effect. Collabra:       Sumner, P. (2018). Low and variable correlation between
    Psychology, 7(1), Article 28202. https://doi.org/10.1525/        reaction time costs and accuracy costs explained by
    collabra.28202                                             accumulation models: Meta-analysis and simulations.

## Page 25

Advances in Methods and Practices in Psychological Science 8(1)                                             25


    Psychological Bulletin, 144(11), 1200–1227. https://doi         social behavior through impacting choice consistency in
    .org/10.1037/bul0000164                                        healthy males. Neuropsychopharmacology, 48(10), Article
Henrich, F., Hartmann, R., Pratz, V., Voss, A., & Klauer, K. C.         10. https://doi.org/10.1038/s41386-023-01570-y
    (2023). The seven-parameter diffusion model: An imple-     LaFollette, K., Fan,  J., Puccio, A., & Demaree, H. A. (2024).
    mentation in stan for Bayesian analyses. Behavior Research       FlexDDM: A flexible decision-diffusion python package for
    Methods, 56, 3102–3116. https://doi.org/10.3758/s13428-        the behavioral sciences. Proceedings of the Annual Meeting
    023-02179-1                                                           of the Cognitive Science Society, 46, 4772–4778. https://
Herz, D. M., Tan, H., Brittain, J.-S., Fischer, P., Cheeran, B.,        escholarship.org/uc/item/4q57r2x0
    Green, A. L., Fitzgerald, J., Aziz, T. Z., Ashkan, K., Little, S.,    Lambert, B. (2018). A student’s guide to Bayesian statistics.
    Foltynie, T., Limousin, P., Zrinzo, L., Bogacz, R., & Brown, P.        Sage.
    (2017). Distinct mechanisms mediate speed-accuracy    Lee, M. D., & Wagenmakers, E.-J. (2014). Bayesian cognitive
    adjustments in cortico-subthalamic networks. eLife, 6,        modeling: A practical course. Cambridge University Press.
    Article e21481. https://doi.org/10.7554/eLife.21481              https://doi.org/10.1017/CBO9781139087759
Hoyer, S., & Hamman,  J. (2017). xarray: N-D labeled arrays    Lerche, V., & Voss, A. (2016). Model complexity in diffusion
   and datasets in python. Journal of Open Research Software,        modeling: Benefits of making the model more parsimoni-
    5(1), 10. https://doi.org/10.5334/jors.148                        ous. Frontiers in Psychology, 7, Article 1324. https://doi
Hu, C.-P., Lan, Y., Macrae, C. N., & Sui, J. (2020). Good me        .org/10.3389/fpsyg.2016.01324
   bad me: Prioritization of the good-self during perceptual    Lerche, V., & Voss, A. (2017). Retest reliability of the parame-
    decision-making. Collabra: Psychology, 6(1), Article 20.          ters of the Ratcliff diffusion model. Psychological Research,
    https://doi.org/10.1525/collabra.301                              81(3), 629–652. https://doi.org/10.1007/s00426-016-0770-5
Johnson, A. A., Ott, M. Q., & Dogucu, M. (2022). Bayes rules!     Liu, Z., Hu, M., Zheng, Y.-R., Sui, J., & Chuan-Peng, H. (2023).
   An introduction to applied Bayesian modeling. Chapman      A multiverse assessment of the reliability of the self match-
   and Hall/CRC. https://www.bayesrulesbook.com/               ing task as a measurement of the self-prioritization effect.
Johnson, D.  J., Hopwood, C.  J., Cesario,  J., & Pleskac, T.  J.        PsyArXiv. https://doi.org/10.31234/osf.io/g6uap
    (2017). Advancing research on cognitive processes in social    Lunn, D., Jackson, C., Best, N., Thomas, A., & Spiegelhalter, D.
   and personality psychology: A hierarchical drift diffusion        (2012). The BUGS book: A practical introduction to
   model primer. Social Psychological and Personality Science,       Bayesian analysis. Chapman and Hall/CRC. https://doi
    8(4), 413–423. https://doi.org/10.1177/1948550617703174        .org/10.1201/b13613
Kass, R. E., & Raftery, A. E. (1995). Bayes factors. Journal of the    Makowski, D., Ben-Shachar, M. S., Chen, S. H. A., & Lüdecke,
   American Statistical Association, 90(430), 773–795. https://       D. (2019). Indices of effect existence and significance
    doi.org/10.1080/01621459.1995.10476572                           in the Bayesian framework. Frontiers in Psychology, 10,
Kelter, R. (2021). Bayesian model selection in the M-open         Article 2767. https://doi.org/10.3389/fpsyg.2019.02767
   setting—Approximate posterior inference and subsam-    Martin, O., Fonnesbeck, C., & Wiecki, T. (2024). Bayesian
    pling for efficient large-scale leave-one-out cross-valida-        analysis with python: A practical guide to probabilistic
    tion via the difference estimator. Journal of Mathematical       modeling (3rd ed.). Packt.
    Psychology, 100, Article 102474. https://doi.org/10.1016/j    Matzke, D., & Wagenmakers, E.-J. (2009). Psychological inter-
    .jmp.2020.102474                                                 pretation of the ex-Gaussian and shifted Wald parameters:
Kelter, R. (2022). fbst: An R package for the full Bayesian signifi-      A diffusion model analysis. Psychonomic Bulletin & Review,
    cance test for testing a sharp null hypothesis against its alter-        16(5), 798–817. https://doi.org/10.3758/PBR.16.5.798
    native via the e value. Behavior Research Methods, 54(3),    McElreath,  R.  (2020).  Statistical rethinking: A Bayesian
    1114–1130. https://doi.org/10.3758/s13428-021-01613-6           course with examples in R and Stan (2nd ed.). Taylor and
Kelter, R. (2023). How to choose between different Bayesian         Francis, CRC Press. https://www.taylorfrancis.com/books/
    posterior indices  for hypothesis  testing  in  practice.        mono/10.1201/9780429029608/statistical-rethinking-rich
    Multivariate Behavioral Research, 58(1), 160–188. https://        ard-mcelreath
    doi.org/10.1080/00273171.2021.1967716                          Mills, J. A. (2018). Objective Bayesian precise hypothesis testing.
Kruschke, J. K. (2014). Doing Bayesian data analysis: A tutorial        University of Cincinnati.
    with R, JAGS, and Stan. Academic Press.                    Pedersen, M. L., Alnæs, D., van der Meer, D., Fernandez-
Kruschke, J. K. (2018). Rejecting or accepting parameter val-        Cabello, S., Berthet, P., Dahl, A., Kjelkenes, R., Schwarz, E.,
   ues in Bayesian estimation. Advances in Methods and       Thompson, W. K., Barch, D. M., Andreassen, O. A., &
    Practices in Psychological Science, 1(2), 270–280. https://        Westlye, L. T. (2022). Computational modeling of the
    doi.org/10.1177/2515245918771304                          N-Back task in the ABCD study: Associations of drift dif-
Kruschke, J. K. (2021). Bayesian analysis reporting guidelines.        fusion model parameters to polygenic scores of mental dis-
   Nature Human Behaviour, 5(10), Article 10. https://doi        orders and cardiometabolic diseases. Biological Psychiatry:
    .org/10.1038/s41562-021-01177-7                                Cognitive Neuroscience and Neuroimaging, 8, 290–299.
Kumar, R., Carroll, C., Hartikainen, A., & Martín, O. A. (2019).        https://doi.org/10.1016/j.bpsc.2022.03.012
    ArviZ: A unified library for exploratory analysis of Bayesian    Pedersen, M. L., & Frank, M. J. (2020). Simultaneous hierar-
   models in python. Journal of Open Source Software, 4(33),         chical Bayesian parameter estimation for reinforcement
    Article 1143. https://doi.org/10.21105/joss.01143                 learning and drift diffusion models: A tutorial and links to
Kutlikova, H. H., Zhang, L., Eisenegger, C., van Honk, J., &        neural data. Computational Brain & Behavior, 3(4), 458–
   Lamm, C. (2023). Testosterone eliminates strategic pro-        471. https://doi.org/10.1007/s42113-020-00084-w

## Page 26

26                                                                               Pan et al.


Pedersen, M. L., Frank, M. J., & Biele, G. (2017). The drift dif-        psychological processes in the diffusion decision model.
    fusion model as the choice rule in reinforcement learning.         Frontiers in Psychology, 11, Article 608287. https://doi.org/
   Psychonomic Bulletin & Review, 24(4), 1234–1251. https://        10.3389/fpsyg.2020.608287
    doi.org/10.3758/s13423-016-1199-y                       van de Schoot, R., Depaoli, S., King, R., Kramer, B., Märtens, K.,
Peikert, A., & Brandmaier, A. M. (2021). A reproducible data        Tadesse, M. G., Vannucci, M., Gelman, A., Veen, D., &
    analysis workflow. Quantitative and Computational        Willemsen,  J. (2021). Bayesian statistics and modelling.
   Methods in Behavioral Sciences, 1, Article e3763. https://       Nature Reviews Methods Primers, 1, Article 16. https://doi
    doi.org/10.5964/qcmb.3763                                    .org/10.1038/s43586-021-00017-2
Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Kothe, U.    van de Schoot,  R., Winter,  S. D., Ryan, O., Zondervan-
    (2022). BayesFlow: Learning complex stochastic models       Zwijnenburg, M., & Depaoli,  S. (2017). A systematic
    with invertible neural networks. IEEE Transactions on       review of Bayesian articles in psychology: The last 25
    Neural Networks and Learning Systems, 33(4), 1452–1466.         years. Psychological Methods, 22(2), 217–239. https://doi
    https://doi.org/10.1109/TNNLS.2020.3042395                   .org/10.1037/met0000100
Ratcliff, R., & McKoon, G. (2008). The diffusion decision model:     Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian
   Theory and data for two-choice decision tasks. Neural       model evaluation using leave-one-out cross-validation and
   Computation, 20(4), 873–922. https://doi.org/10.1162/       WAIC. Statistics and Computing, 27(5), 1413–1432. https://
    neco.2008.12-06-420                                           doi.org/10.1007/s11222-016-9696-4
Ratcliff, R., Smith, P. L., Brown, S. D., & McKoon, G. (2016).     Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner,
    Diffusion decision model: Current issues and history.         P.-C. (2021). Rank-normalization, folding, and localization:
    Trends in Cognitive Sciences, 20(4), 260–281. https://doi      An improved R2 for assessing convergence of MCMC (with
    .org/10.1016/j.tics.2016.01.007                                     discussion). Bayesian Analysis, 16(2), 667–718. https://doi
Ratcliff, R., & Tuerlinckx, F. (2002). Estimating parameters of        .org/10.1214/20-BA1221
    the diffusion model: Approaches to dealing with contami-    Voss, A., Nagler, M., & Lerche, V. (2013). Diffusion models in
    nant reaction times and parameter variability. Psychonomic        experimental psychology. Experimental Psychology, 60(6),
    Bulletin & Review, 9(3), 438–481. https://doi.org/10.3758/        385–402. https://doi.org/10.1027/1618-3169/a000218
    bf03196302                                                 Voss, A., & Voss, J. (2007). Fast-dm: A free program for effi-
Robert, C. P., & Casella, G. (2004). The Metropolis—Hastings         cient diffusion model analysis. Behavior Research Methods,
    algorithm. In C. P. Robert & G. Casella (Eds.), Monte Carlo        39(4), 767–775. https://doi.org/10.3758/BF03192967
     statistical methods (pp. 267–320). Springer. https://doi.    Wabersich, D., & Vandekerckhove,  J. (2014). Extending
    org/10.1007/978-1-4757-4145-2_7                             JAGS: A tutorial on adding custom distributions to JAGS
Schubert, A.-L., Hagemann, D., Voss, A., & Bergmann, K.        (with a diffusion model example). Behavior Research
    (2017). Evaluating the model fit of diffusion models with        Methods, 46(1), 15–28. https://doi.org/10.3758/s13428-
    the root mean square error of approximation. Journal of        013-0369-3
   Mathematical Psychology, 77, 29–45. https://doi.org/10    Wagenmakers, E.-J., Lodewyckx, T., Kuriyal, H., & Grasman, R.
    .1016/j.jmp.2016.08.004                                          (2010). Bayesian hypothesis testing for psychologists:
Shadlen, M. N., & Shohamy, D. (2016). Decision making and      A  tutorial on the Savage–Dickey method. Cognitive
    sequential sampling from memory. Neuron, 90(5), 927–        Psychology, 60(3), 158–189. https://doi.org/10.1016/j.cog
    939. https://doi.org/10.1016/j.neuron.2016.04.036               psych.2009.12.001
Sheng, F., Ramakrishnan, A., Seok, D., Zhao, W. J., Thelaus, S.,    Wagenmakers,  E.-J., Van Der Maas, H. L.  J., & Grasman,
    Cen, P., & Platt, M. L. (2020). Decomposing loss aversion         R. P. P. P. (2007). An EZ-diffusion model for response time
   from gaze allocation and pupil dilation. Proceedings of the       and accuracy. Psychonomic Bulletin & Review, 14(1), 3–22.
    National Academy of Sciences, USA, 117(21), 11356–11363.        https://doi.org/10.3758/BF03194023
    https://doi.org/10.1073/pnas.1919670117                  Watanabe, S. (2010). Asymptotic equivalence of Bayes cross
Shinn, M., Lam, N. H., & Murray, J. D. (2020). A flexible frame-         validation and widely applicable information criterion in
   work for simulating and fitting generalized drift-diffusion        singular learning theory. Journal of Machine Learning
    models. eLife, 9, 1–27. https://doi.org/10.7554/elife.56938        Research, 11(12), 3571–3594.
Singmann, H., Brown, S., Gretton, M., Heathcote, A., Voss, A.,    Wiebels, K., & Moreau, D. (2021). Leveraging containers for
    Voss,  J., & Terry, A. (2022). rtdists: Response time dis-        reproducible psychological research. Advances in Methods
    tributions (Version 0.11-5) [Computer software]. https://      and Practices in Psychological Science, 4(2). https://doi
    10.32614/CRAN.package.rtdists                                .org/10.1177/25152459211017853
Spiegelhalter, D.  J., Best, N. G., Carlin, B. P., & Van Der    Wiecki, T.  V., Sofer,  I., & Frank, M.  J. (2013). HDDM:
    Linde, A. (2002). Bayesian measures of model complex-        Hierarchical Bayesian estimation of the drift-diffusion
     ity and  fit. Journal of the Royal Statistical Society Series       model in python. Frontiers in Neuroinformatics, 7, Article
    B: Statistical Methodology, 64(4), 583–639. https://doi         14. https://doi.org/10.3389/fninf.2013.00014
    .org/10.1111/1467-9868.00353                              Wilson, R. C., & Collins, A. G. (2019). Ten simple rules for the
Stevenson, N., Donzallaz, M. C., Innes, R., Forstmann, B.,        computational modeling of behavioral data. eLife, 8, Article
    Matzke, D., & Heathcote, A. (2024). EMC2: An R package        e49547. https://doi.org/10.7554/eLife.49547
    for cognitive models of choice. PsyArXiv. https://doi.org/    Zhang, L., Lengersdorff, L., Mikus, N., Glascher, J., & Lamm,
    10.31234/osf.io/2e4dq                                            C. (2020). Using reinforcement learning models in social
Tran, N.-H., Van Maanen, L., Heathcote, A., & Matzke, D.        neuroscience: Frameworks, pitfalls and suggestions of
    (2021). Systematic parameter reviews in cognitive model-        best practices. Social Cognitive and Affective Neuroscience,
    ing: Towards a robust and cumulative characterization of        15(6), 695–707. https://doi.org/10.1093/scan/nsaa089
