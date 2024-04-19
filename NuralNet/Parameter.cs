﻿namespace NN;

public class Parameter
{
    public double Value { get; set; }

    public double TempGradient { get; set; } = 0;

    public bool IsOptimized { get; set; }
}