from libraries import *

df = pd.read_csv('../code/final_df.csv')
print(df.shape)   # (38496, 11)

sns.set(font_scale=1.2)
sns.set_theme(style='whitegrid')

'''Number of masks of each image'''
df_bar = df['counts'].value_counts().to_frame().reset_index()
df_bar.columns = ['Number of masks', 'Count']
df_bar = df_bar.sort_values(by='Count', ascending=False)

sns.barplot(data=df_bar,
            x='Number of masks',
            y='Count',
            order=df_bar.sort_values('Count', ascending=False)['Number of masks'])
plt.title('Count - Number of Masks in each Image')
plt.tight_layout()

for i, v in enumerate(df_bar.Count):
    plt.text(i, v, f"{v}", ha='center')

plt.show()


'''Percentages of Images with Masks'''
organs = ['Large Bowel', 'Small Bowel', 'Stomach']
N = df.iloc[:, 1:4].notna().sum().values.tolist()
percent = [round(n/df.shape[0]*100, 2) for n in N]

sns.barplot(x=organs,
            y=percent)
plt.title('Percentages - Images with Masks')
plt.xlabel('Organ')
plt.ylabel('Percentages')
plt.ylim(0, 50)

for i, v in enumerate(percent):
    plt.text(i, v+1, f"{v}%", ha='center')

plt.show()