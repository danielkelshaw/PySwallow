class PlotDesigner:

    """Containts information required to format plots."""

    def __init__(self):

        """
        Initialises the PlotDesigner Class.

        Attributes
        ----------
        figsize : tuple
            Figsize in inches for the plot.
        title_fontsize : str, int
            Matplotlib command for the title fontsize.
        text_fontsize : str, int
            Matplotlib command for the text fontsize.
        label : list
            Labels for each of the respective axes.
        limit : list
            Tuples descirbing the limits for the axes.
        """

        self.figsize = (10, 8)
        self.title_fontsize = 'large'
        self.text_fontsize = 'medium'
        self.label = ['x-axis', 'y-axis']
        self.limit = [(-1, 1), (-1, 1)]
