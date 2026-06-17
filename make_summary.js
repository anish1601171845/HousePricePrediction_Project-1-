const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
        VerticalAlign } = require('docx');
const fs = require('fs');

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1E3A5F" },
        paragraph: { spacing: { before: 300, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2E6DA4" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
      }
    },
    children: [
      // Title block
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 100 },
        children: [new TextRun({ text: "🏠 House Price Prediction", bold: true, size: 44, font: "Arial", color: "1E3A5F" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "Internship Project — Week 1  |  Summary Report", size: 24, font: "Arial", color: "555555" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "Dataset: Kaggle Housing Prices Dataset (545 rows × 13 columns)", size: 20, italics: true, color: "777777" })]
      }),

      // Section 1
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Objective")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "Build a machine learning regression pipeline to predict house prices from property features, evaluate model performance, and identify which features most strongly drive pricing.", size: 22 })]
      }),

      // Section 2 - Dataset
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. Dataset Overview")] }),
      new Paragraph({ spacing: { after: 140 }, children: [new TextRun({ text: "The Housing Prices Dataset contains 545 property records with 13 columns:", size: 22 })] }),

      // Feature table
      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [
          new TableRow({
            tableHeader: true,
            children: [
              new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: "Column", bold: true, color: "FFFFFF" })] })],
                shading: { fill: "2E6DA4", type: ShadingType.CLEAR }, width: { size: 35, type: WidthType.PERCENTAGE } }),
              new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: "Type", bold: true, color: "FFFFFF" })] })],
                shading: { fill: "2E6DA4", type: ShadingType.CLEAR }, width: { size: 25, type: WidthType.PERCENTAGE } }),
              new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: "Description", bold: true, color: "FFFFFF" })] })],
                shading: { fill: "2E6DA4", type: ShadingType.CLEAR }, width: { size: 40, type: WidthType.PERCENTAGE } }),
            ]
          }),
          ...[ 
            ["price", "Numeric (Target)", "House selling price"],
            ["area", "Numeric", "Property area in sq ft"],
            ["bedrooms", "Numeric", "Number of bedrooms"],
            ["bathrooms", "Numeric", "Number of bathrooms"],
            ["stories", "Numeric", "Number of storeys"],
            ["mainroad", "Binary (yes/no)", "Fronts a main road"],
            ["guestroom", "Binary (yes/no)", "Has a guest room"],
            ["basement", "Binary (yes/no)", "Has a basement"],
            ["hotwaterheating", "Binary (yes/no)", "Has hot water heating"],
            ["airconditioning", "Binary (yes/no)", "Has air conditioning"],
            ["parking", "Numeric", "Number of parking spots"],
            ["prefarea", "Binary (yes/no)", "In a preferred area"],
            ["furnishingstatus", "Categorical", "Furnished / Semi / Unfurnished"],
          ].map((row, i) => new TableRow({
            children: row.map(text => new TableCell({
              shading: i % 2 === 0 ? { fill: "F0F7FF", type: ShadingType.CLEAR } : undefined,
              children: [new Paragraph({ children: [new TextRun({ text, size: 20 })] })]
            }))
          }))
        ]
      }),

      new Paragraph({ spacing: { before: 300 } }),

      // Section 3 - Cleaning
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Data Cleaning Steps")] }),
      ...[
        "No missing values were found in the dataset.",
        "No duplicate rows were present.",
        "Binary yes/no columns (mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea) were encoded as 0/1.",
        "The furnishingstatus column (3 categories) was one-hot encoded, with drop_first=True to avoid multicollinearity.",
      ].map(text => new Paragraph({
        spacing: { after: 100 },
        bullet: { level: 0 },
        children: [new TextRun({ text, size: 22 })]
      })),

      new Paragraph({ spacing: { before: 200 } }),

      // Section 4 - Models
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Model Performance")] }),
      new Paragraph({ spacing: { after: 140 }, children: [new TextRun({ text: "Two regression models were trained on an 80/20 train-test split:", size: 22 })] }),

      new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [
          new TableRow({
            tableHeader: true,
            children: ["Model", "MAE", "RMSE", "R² Score"].map(h =>
              new TableCell({
                shading: { fill: "1E3A5F", type: ShadingType.CLEAR },
                children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 22 })] })]
              })
            )
          }),
          new TableRow({
            children: [
              ["Linear Regression", "579,414", "727,287", "0.8513 ✓ Best"].map((t, i) =>
                new TableCell({
                  shading: { fill: "E8F5E9", type: ShadingType.CLEAR },
                  children: [new Paragraph({ alignment: i > 0 ? AlignmentType.CENTER : AlignmentType.LEFT, children: [new TextRun({ text: t, size: 22, bold: i === 3 })] })]
                })
              )
            ].flat()
          }),
          new TableRow({
            children: [
              ["Random Forest", "685,023", "833,430", "0.8048"].map((t, i) =>
                new TableCell({
                  children: [new Paragraph({ alignment: i > 0 ? AlignmentType.CENTER : AlignmentType.LEFT, children: [new TextRun({ text: t, size: 22 })] })]
                })
              )
            ].flat()
          }),
        ]
      }),

      new Paragraph({ spacing: { before: 200 } }),

      // Section 5 - Insights
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. Key Insights")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Most Influential Features")] }),
      new Paragraph({
        spacing: { after: 160 },
        children: [new TextRun({ text: "Area (sq ft) is the single strongest predictor of house price, accounting for approximately 76% of feature importance in the Random Forest model. Bathrooms (6.1%), stories (3.3%), and parking spaces follow as the next most impactful features. Amenity features like air conditioning and preferred area location contribute moderately, while guest room and hot water heating have minimal impact.", size: 22 })]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Model Accuracy")] }),
      new Paragraph({
        spacing: { after: 160 },
        children: [new TextRun({ text: "Linear Regression achieved the best performance with R² = 0.85, meaning the model explains 85% of the variation in house prices. This is a strong result and suggests that price relationships in this dataset are largely linear. Random Forest (R² = 0.80) performed slightly worse, possibly due to the limited size of the dataset reducing its advantage over simpler models.", size: 22 })]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Surprising Finding")] }),
      new Paragraph({
        spacing: { after: 160 },
        children: [new TextRun({ text: "The dominance of area as a predictor was striking — its importance dwarfed all other features combined. Also unexpected was that furnishing status had a relatively minor effect on price, suggesting buyers prioritize structural size and amenities over interior presentation.", size: 22 })]
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("Business Recommendation")] }),
      new Paragraph({
        spacing: { after: 200 },
        children: [new TextRun({ text: "Real estate developers and agents should prioritize maximizing usable floor area and bathroom count when investing in properties. These features yield the strongest price premiums. For premium-segment properties, adding air conditioning and parking spaces offer high ROI, while furniture investment alone is unlikely to substantially increase market value.", size: 22 })]
      }),

      // Footer line
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 300 },
        children: [new TextRun({ text: "House Price Prediction Project  |  Internship Week 1  |  Generated with Python & Scikit-learn", size: 18, italics: true, color: "999999" })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('/home/claude/HousePricePrediction/summary.docx', buffer);
  console.log('summary.docx created!');
});
